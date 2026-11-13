"""独立英文问候 SFT，复用已有基座与训练器，不联网。"""

import copy
import gc
import json
from pathlib import Path
import re
import tempfile

import torch

from src.alignment_math import response_batch
from src.alignment_tasks import dev_loss, model_from_checkpoint, read_examples, train_stage
from src.data import file_hash, write_json
from src.progress import progress_bar
from src.science_data import fingerprint
from src.tokenizer import BPETokenizer
from src.training import amp, setup


def dialogue_prompt(history, message):
    prefix = "".join(f"{'User' if row['role'] == 'user' else 'Assistant'}: {row['content']}\n" for row in history)
    return prefix + f"User: {message}\nAssistant:"


def fit_history(tokenizer, context, history, message, max_new_tokens):
    if not isinstance(history, list) or len(history) > 20 or len(history) % 2:
        raise ValueError("聊天历史应为至多10轮完整的用户/助手消息")
    for index, row in enumerate(history):
        if (not isinstance(row, dict) or row.get("role") != ("user" if index % 2 == 0 else "assistant") or
                not isinstance(row.get("content"), str) or not row["content"].strip() or
                (row["role"] == "user" and not row["content"].isascii()) or len(row["content"]) > 1200):
            raise ValueError("聊天历史格式无效；仅支持英文消息")
    if not isinstance(message, str) or not message.strip() or not message.isascii() or len(message) > 1200:
        raise ValueError("请输入简短的英文消息，例如 Hello! 或 How are you?")
    kept = [{"role":row["role"], "content":" ".join(row["content"].split())} for row in history]
    message = " ".join(message.split())
    while True:
        prompt = dialogue_prompt(kept, message)
        if len(tokenizer.encode(prompt)) + max_new_tokens - 1 <= context:
            return prompt, len(kept) // 2
        if not kept:
            raise ValueError("这条消息超出模型的小上下文窗口，请缩短后再发送")
        kept = kept[2:]


@torch.no_grad()
def chat_reply(model, tokenizer, prompt, max_new_tokens, device, precision):
    ids = tokenizer.encode(prompt)
    if not ids or len(ids) + max_new_tokens - 1 > model.context:
        raise ValueError("聊天消息和回答超过上下文")
    was_training = model.training
    model.eval()
    response, stop = [], "length"
    try:
        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with amp(device, precision):
                logits, _ = model(x)
            scores = logits[0, -1].float()
            scores[[tokenizer.pad_id, tokenizer.bos_id]] = -float("inf")
            token = int(scores.argmax())
            response.append(token)
            if token == tokenizer.eos_id:
                stop = "eos"
                break
            ids.append(token)
            text = tokenizer.decode(response, errors="replace")
            if "\nUser:" in text or "\nAssistant:" in text:
                stop = "role_boundary"
                break
        raw = tokenizer.decode(response, errors="replace")
        text = re.split(r"\n(?:User|Assistant):", raw, maxsplit=1)[0].strip()
    finally:
        model.train(was_training)
    return dict(text=text, new_tokens=len(response), stop_reason=stop)


def prepare_chat(cfg):
    tokenizer_path = Path(cfg["tokenizer_dir"]) / "tokenizer.json"
    instance = BPETokenizer.load(tokenizer_path)
    source = Path(cfg["dataset"])
    seed_data = json.loads(source.read_text(encoding="utf-8"))
    provenance = dict(version=1, dataset_sha256=file_hash(source), tokenizer_sha256=file_hash(tokenizer_path),
                      max_prompt_tokens=cfg["max_prompt_tokens"], max_new_tokens=cfg["max_new_tokens"])
    signature = fingerprint(provenance)
    root = Path(cfg["output_dir"]).resolve() / "data"
    if root.exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("英文对话数据配置已改变，请使用新的 chat.output_dir")
        for split, info in manifest["files"].items():
            if file_hash(root / f"{split}.jsonl") != info["sha256"]:
                raise ValueError("英文对话数据校验失败")
        return root, manifest, dict(instance=instance, sha256=provenance["tokenizer_sha256"])
    rows = {split:[] for split in ("train", "dev", "test")}
    for split in rows:
        for intent in seed_data["intents"]:
            answers = intent["answers"] if split == "train" else intent["answers"][:1]
            for query in intent[split]:
                for answer in answers:
                    for history in ([], [dict(role="user", content="Hello!"), dict(role="assistant", content="Hello! How can I help?")]):
                        rows[split].append(dict(prompt=dialogue_prompt(history, query), chosen=answer,
                                                intent=intent["name"], expected_name=None))
        for name in seed_data["memory_names"][split]:
            query, answer = f"My name is {name}.", f"Nice to meet you, {name}!"
            rows[split].append(dict(prompt=dialogue_prompt([], query), chosen=answer, intent="introduction", expected_name=name))
            history = [dict(role="user", content=query), dict(role="assistant", content=answer)]
            rows[split].append(dict(prompt=dialogue_prompt(history, "What is my name?"), chosen=f"Your name is {name}.",
                                    intent="context_name", expected_name=name))
    normalized = {}
    for split, values in rows.items():
        normalized[split] = {" ".join(row["prompt"].lower().split()) for row in values}
        for row in values:
            if len(instance.encode(row["prompt"])) > cfg["max_prompt_tokens"] or len(instance.encode(row["chosen"], add_eos=True)) > cfg["max_new_tokens"]:
                raise ValueError("对话样本超出配置长度，请增大长度上限或缩短样本")
    if normalized["train"] & (normalized["dev"] | normalized["test"]) or normalized["dev"] & normalized["test"]:
        raise ValueError("对话训练/开发/测试的完整 prompt 重叠")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".chat-data-", dir=root.parent) as directory:
        staging = Path(directory) / "data"
        staging.mkdir()
        files = {}
        for split, values in rows.items():
            path = staging / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
            files[split] = dict(examples=len(values), sha256=file_hash(path))
        manifest = dict(signature=signature, provenance=provenance, files=files,
                        description=seed_data["provenance"], split_rule="manually separated query paraphrases and memory names; normalized full prompts checked disjoint")
        write_json(staging / "manifest.json", manifest)
        staging.rename(root)
    return root, manifest, dict(instance=instance, sha256=provenance["tokenizer_sha256"])


def run_chat(base_cfg, task):
    cfg = base_cfg["chat"]
    if type(cfg["seed"]) is not int or cfg["seed"] < 0 or (cfg["stop_after"] is not None and (type(cfg["stop_after"]) is not int or cfg["stop_after"] < 1)):
        raise ValueError("chat.seed/stop_after 无效")
    for key in ("batch_size", "grad_accum", "max_prompt_tokens", "max_new_tokens", "eval_every", "save_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"chat.{key} 必须为正整数")
    sft = cfg["sft"]
    if (type(sft["steps"]) is not int or sft["steps"] < 1 or not 0 < sft["min_lr"] <= sft["lr"] or
            type(sft["warmup_steps"]) is not int or not 0 <= sft["warmup_steps"] < sft["steps"]):
        raise ValueError("chat.sft 预算或学习率无效")
    root, manifest, tokenizer = prepare_chat(cfg)
    rows = {split:read_examples(root / f"{split}.jsonl") for split in ("train", "dev", "test")}
    if task == "chat_prepare":
        return dict(prepared=True, output_dir=str(root.parent), dataset=manifest)
    options = dict(base_cfg["alignment"], **{key:cfg[key] for key in ("output_dir", "seed", "batch_size", "grad_accum",
                    "max_prompt_tokens", "max_new_tokens", "eval_every", "save_every", "stop_after", "sft")}, min_words=0)
    parent = Path(cfg["checkpoint"])
    if not parent.is_file():
        raise FileNotFoundError("未找到已有故事模型，请检查 chat.checkpoint")
    if task == "chat":
        path, completed = train_stage(base_cfg, options, "sft", parent, manifest, tokenizer, rows["train"], rows["dev"])
        if not completed:
            return dict(completed=False, message="英文对话 SFT 断点已保存，续训设 chat.stop_after=null")
    else:
        path = root.parent / "sft" / "final.pt"
    setup_cfg = copy.deepcopy(base_cfg)
    setup_cfg["train"]["seed"] = cfg["seed"]
    device, precision = setup(setup_cfg)
    evaluation = []
    for name, checkpoint in (("base", parent), ("chat", path)):
        model = model_from_checkpoint(checkpoint, tokenizer, device).eval()
        loss = dev_loss(model, None, rows["test"], tokenizer["instance"], options, device, precision, "sft")
        samples = []
        for row in progress_bar(rows["test"], config=base_cfg["progress"], desc=f"Chat eval {name}", unit="task"):
            result = chat_reply(model, tokenizer["instance"], row["prompt"], cfg["max_new_tokens"], device, precision)
            expected_name = row["expected_name"]
            samples.append(dict(prompt=row["prompt"], reference=row["chosen"], intent=row["intent"],
                                expected_name=expected_name, name_match=bool(re.search(r"\b"+re.escape(expected_name)+r"\b", result["text"], re.I)) if expected_name else None,
                                **result))
        names = [row["name_match"] for row in samples if row["intent"] == "context_name"]
        evaluation.append(dict(model=name, test_loss=loss, empty_response_rate=sum(not r["text"] for r in samples)/len(samples),
                               context_name_match_rate=sum(names)/len(names) if names else None, samples=samples))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = dict(completed=True, dataset=manifest, evaluation=evaluation,
                  limitations=["项目内编写的合成种子模板和自动展开样本，非大型公开聊天数据集。",
                               "训练/测试覆盖相同意图但不同问法；多轮名字只用于窄范围记忆检查。",
                               "没有自动泛化聊天准确率，测试 loss 与名字匹配不能代替人工阅读。",
                               "只覆盖英文问候、感谢、自我介绍和短交流；不保证复杂问答。"])
    write_json(root.parent / "report.json", report)
    from src.report_tasks import export_report
    project_report = export_report(base_cfg)
    return dict(completed=True, output_dir=str(root.parent), checkpoint=str(path), report=str(root.parent / "report.json"),
                project_report=project_report["report"],
                examples={k:len(v) for k,v in rows.items()}, evaluation=[{k:v for k,v in row.items() if k != "samples"} for row in evaluation])

"""同一故事基座上的 SFT / DPO / 可验证奖励微调；不调用外部 API。"""

import copy
import csv
import gc
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.alignment_data import prepare_alignment, reward
from src.alignment_math import dpo_loss, reinforce_loss, response_batch, response_log_probs
from src.data import file_hash, write_json
from src.model import GPT
from src.progress import progress_bar
from src.science_data import fingerprint
from src.science_tasks import atomic_checkpoint
from src.training import amp, append_json, learning_rate, load_checkpoint, setup, synchronize


def validate_alignment(cfg):
    for key in ("train_examples", "dev_examples", "test_examples", "max_scan", "max_new_tokens",
                "min_words", "max_prompt_tokens", "batch_size", "grad_accum", "eval_every", "save_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"alignment.{key} 必须为正整数")
    if cfg["stop_after"] is not None and (type(cfg["stop_after"]) is not int or cfg["stop_after"] < 1):
        raise ValueError("alignment.stop_after 必须为正整数或 null")
    for key in ("seed", "evaluation_seed"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError("alignment 种子必须为非负整数")
    if (len(set(cfg["keywords"])) != len(cfg["keywords"]) or len(cfg["keywords"]) < 4 or
            any(not isinstance(word, str) or not word.isascii() or not word.isalpha() or word != word.lower() for word in cfg["keywords"])):
        raise ValueError("keywords 需要至少四个不重复的小写英文单词")
    for name in ("sft", "dpo", "rlvr"):
        options = cfg[name]
        if type(options["steps"]) is not int or options["steps"] < 1 or not 0 < options["min_lr"] <= options["lr"]:
            raise ValueError(f"alignment.{name} 步数或学习率无效")
        if type(options["warmup_steps"]) is not int or not 0 <= options["warmup_steps"] < options["steps"]:
            raise ValueError(f"alignment.{name}.warmup_steps 无效")
    if cfg["dpo"]["beta"] <= 0 or cfg["rlvr"]["kl_beta"] < 0:
        raise ValueError("DPO beta 必须正，RLVR kl_beta 必须非负")
    if type(cfg["rlvr"]["group_size"]) is not int or cfg["rlvr"]["group_size"] < 2:
        raise ValueError("RLVR group_size 至少为 2")


def read_examples(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def model_from_checkpoint(path, tokenizer, device):
    checkpoint = load_checkpoint(path)
    if checkpoint["tokenizer_sha256"] != tokenizer["sha256"]:
        raise ValueError("模型权重与当前分词器不一致")
    config = dict(checkpoint["model_config"], dropout=0.0)
    if config["vocab_size"] != tokenizer["instance"].vocab_size:
        raise ValueError("权重与词表大小不同")
    model = GPT(**config)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    return model.to(device)


@torch.no_grad()
def sample_responses(model, tokenizer, prompt, count, max_new_tokens, generator, device, precision):
    prefix = tokenizer.encode(prompt)
    if not prefix or len(prefix) + max_new_tokens - 1 > model.context:
        raise ValueError("约束生成的 prompt + 回答超过模型上下文")
    was_training = model.training
    model.eval()
    ids = torch.tensor([prefix] * count, dtype=torch.long, device=device)
    responses = [[] for _ in range(count)]
    active = [True] * count
    try:
        for _ in range(max_new_tokens):
            with amp(device, precision):
                logits, _ = model(ids)
            scores = logits[:, -1].float()
            scores[:, [tokenizer.pad_id, tokenizer.bos_id]] = -float("inf")
            tokens = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            values = tokens[:, 0].tolist()
            for row, token in enumerate(values):
                if active[row]:
                    responses[row].append(token)
                    active[row] = token != tokenizer.eos_id
            if not any(active) or ids.shape[1] == model.context:
                break
            ids = torch.cat((ids, tokens), dim=1)
    finally:
        model.train(was_training)
    return responses


def chosen_rejected_batch(examples, tokenizer, context, device):
    chosen = [dict(example, chosen=example["chosen"]) for example in examples]
    rejected = [dict(example, chosen=example["rejected"]) for example in examples]
    return response_batch(chosen + rejected, tokenizer, context, device)


@torch.no_grad()
def dev_loss(model, reference, rows, tokenizer, cfg, device, precision, method):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), cfg["batch_size"]):
        examples = rows[start:start + cfg["batch_size"]]
        if method == "sft":
            x, y = response_batch(examples, tokenizer, model.context, device)
            with amp(device, precision):
                logits, _ = model(x)
            total += float(F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"))
            count += int((y != -100).sum())
        else:
            x, y = chosen_rejected_batch(examples, tokenizer, model.context, device)
            with amp(device, precision):
                ref_logits, _ = reference(x)
                logits, _ = model(x)
            ref, _, _ = response_log_probs(ref_logits, y)
            scores, _, _ = response_log_probs(logits, y)
            size = len(examples)
            value, _ = dpo_loss(scores[:size], scores[size:], ref[:size], ref[size:], cfg["dpo"]["beta"])
            total += float(value) * size
            count += size
    return total / count


def train_stage(base_cfg, cfg, method, parent_path, manifest, tokenizer, train_rows, dev_rows):
    options = copy.deepcopy(base_cfg)
    stage = cfg[method]
    options["train"].update(seed=cfg["seed"], max_steps=stage["steps"], warmup_steps=stage["warmup_steps"],
                            lr=stage["lr"], min_lr=stage["min_lr"])
    device, precision = setup(options)
    signature = fingerprint(dict(version=1, method=method, parent_sha256=file_hash(parent_path),
                                 data=manifest["signature"], stage=stage, seed=cfg["seed"],
                                 batch=cfg["batch_size"], accum=cfg["grad_accum"], precision=precision,
                                 max_new_tokens=cfg["max_new_tokens"], min_words=cfg["min_words"],
                                 torch=str(torch.__version__), device=device.type,
                                 optimizer={key:options["train"][key] for key in ("beta1", "beta2", "grad_clip")}))
    output = Path(cfg["output_dir"]).resolve() / method
    output.mkdir(parents=True, exist_ok=True)
    result_path, final_path, last_path = (output / name for name in ("result.json", "final.pt", "last.pt"))
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature or not final_path.is_file() or file_hash(final_path) != result["checkpoint_sha256"]:
            raise ValueError(f"{method} 已完成结果不匹配，请使用新 alignment.output_dir")
        logging.info("跳过已完成阶段 %s", method)
        return final_path, True
    if any(output.iterdir()) and not last_path.exists():
        raise ValueError(f"{method} 目录不完整且没有断点，请检查输出目录")
    model = model_from_checkpoint(parent_path, tokenizer, device)
    if cfg["max_prompt_tokens"] + cfg["max_new_tokens"] - 1 > model.context:
        raise ValueError("alignment 的指令与回答长度上限之和超过上下文")
    reference = None
    if method != "sft":
        reference = model_from_checkpoint(parent_path, tokenizer, device).eval()
        reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage["lr"],
                                 betas=(options["train"]["beta1"], options["train"]["beta2"]), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    rng = torch.Generator().manual_seed(cfg["seed"])
    sampler = torch.Generator(device=device).manual_seed(cfg["seed"] + 123)
    start, elapsed, generated, zero_groups = 0, 0.0, 0, 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError(f"{method} 断点来源或配置不匹配")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.set_state(state["rng"])
        sampler.set_state(state["sampler_rng"])
        start, elapsed, generated, zero_groups = state["step"], state["seconds"], state["generated_tokens"], state["zero_groups"]
        del state

    def save(step):
        atomic_checkpoint(last_path, dict(version=1, signature=signature, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng.get_state(),
                          sampler_rng=sampler.get_state(), step=step, seconds=elapsed,
                          generated_tokens=generated, zero_groups=zero_groups))

    if not last_path.exists():
        save(0)
    end = min(stage["steps"], cfg["stop_after"] or stage["steps"])
    if end <= start and end < stage["steps"]:
        raise ValueError("stop_after 已达到，续训请设置 alignment.stop_after=null")
    logging.info("%s：从 step %d 训练到 %d，来源 %s", method, start, stage["steps"], parent_path)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with progress_bar(total=stage["steps"], initial=start, config=base_cfg["progress"], desc=method.upper(), unit="step") as bar:
        for step in range(start, end):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, dict(stage, max_steps=stage["steps"]))
            model.train()
            synchronize(device)
            began = time.perf_counter()
            record = dict(step=step + 1)
            if method == "rlvr":
                example = train_rows[int(torch.randint(len(train_rows), (1,), generator=rng))]
                completions = sample_responses(model, tokenizer["instance"], example["prompt"], stage["group_size"],
                                               cfg["max_new_tokens"], sampler, device, precision)
                values = [reward(tokenizer["instance"].decode(ids, errors="replace"), example["words"], cfg["min_words"])["reward"]
                          for ids in completions]
                generated += sum(map(len, completions))
                x, y = response_batch([example] * len(completions), tokenizer["instance"], model.context, device, completions)
                with torch.no_grad(), amp(device, precision):
                    ref_logits, _ = reference(x)
                with amp(device, precision):
                    logits, _ = model(x)
                loss, diagnostics = reinforce_loss(logits, ref_logits, y, torch.tensor(values, device=device),
                                                   stage["kl_beta"], (tokenizer["instance"].pad_id, tokenizer["instance"].bos_id))
                zero_groups += int(diagnostics["zero_advantage_group"])
                record.update(diagnostics)
                scaler.scale(loss).backward()
            else:
                batches = []
                for _ in range(cfg["grad_accum"]):
                    indices = torch.randint(len(train_rows), (cfg["batch_size"],), generator=rng).tolist()
                    examples = [train_rows[index] for index in indices]
                    builder = response_batch if method == "sft" else chosen_rejected_batch
                    batches.append(builder(examples, tokenizer["instance"], model.context, device))
                total_targets = sum(int((y != -100).sum()) for _, y in batches)
                loss_value = 0.0
                for x, y in batches:
                    ref_scores = None
                    if method == "dpo":
                        with torch.no_grad(), amp(device, precision):
                            ref_logits, _ = reference(x)
                        ref_scores, _, _ = response_log_probs(ref_logits, y)
                        del ref_logits
                    with amp(device, precision):
                        logits, _ = model(x)
                    if method == "sft":
                        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum") / total_targets
                    else:
                        scores, _, _ = response_log_probs(logits, y)
                        size = x.shape[0] // 2
                        loss, margin = dpo_loss(scores[:size], scores[size:], ref_scores[:size], ref_scores[size:], stage["beta"])
                        loss = loss / cfg["grad_accum"]
                        record["preference_margin"] = float(margin.mean())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("微调 loss 非有限值")
                    loss_value += float(loss.detach())
                    scaler.scale(loss).backward()
                loss = torch.tensor(loss_value)
            if not torch.isfinite(loss):
                raise FloatingPointError("微调 loss 非有限值")
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["train"]["grad_clip"])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("微调梯度非有限值")
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
            elapsed += time.perf_counter() - began
            record.update(loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"])
            if method != "rlvr" and ((step + 1) % cfg["eval_every"] == 0 or step + 1 == end):
                record["dev_loss"] = dev_loss(model, reference, dev_rows, tokenizer["instance"], cfg, device, precision, method)
            append_json(output / "metrics.jsonl", record)
            if (step + 1) % cfg["save_every"] == 0 or step + 1 == end:
                save(step + 1)
            bar.set_postfix(loss=f"{record['loss']:.4f}", refresh=False)
            bar.update(1)
    if end < stage["steps"]:
        return last_path, False
    # 保存与基座一致的结构信息；dropout 固定为 0，使采样策略与评分策略一致。
    parent = load_checkpoint(parent_path)
    model_cfg = dict(parent["model_config"], dropout=0.0)
    del parent
    atomic_checkpoint(final_path, dict(version=1, signature=signature, model=model.state_dict(), model_config=model_cfg,
                      tokenizer_sha256=tokenizer["sha256"], stage=method, step=stage["steps"]))
    result = dict(signature=signature, method=method, completed=True, checkpoint_sha256=file_hash(final_path),
                  steps=stage["steps"], train_seconds=elapsed, generated_tokens=generated, zero_advantage_groups=zero_groups,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None,
                  algorithm="group-relative on-policy REINFORCE + categorical KL" if method == "rlvr" else method.upper(),
                  seed=cfg["seed"], precision=precision, torch=str(torch.__version__))
    write_json(result_path, result)
    del model, reference, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_path, True


def evaluate_checkpoint(base_cfg, cfg, name, path, tokenizer, test_rows, data_signature):
    options = copy.deepcopy(base_cfg)
    options["train"]["seed"] = cfg["evaluation_seed"]
    device, precision = setup(options)
    signature = fingerprint(dict(version=1, checkpoint=file_hash(path), data=data_signature, precision=precision,
                                 max_new_tokens=cfg["max_new_tokens"], min_words=cfg["min_words"],
                                 evaluation_seed=cfg["evaluation_seed"]))
    output = Path(cfg["output_dir"]).resolve() / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{name}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] == signature:
            return result
    model = model_from_checkpoint(path, tokenizer, device).eval()
    samples, began = [], time.perf_counter()
    for index, row in enumerate(progress_bar(test_rows, config=base_cfg["progress"], desc=f"Evaluate {name}", unit="task")):
        rng = torch.Generator(device=device).manual_seed(cfg["evaluation_seed"] + index)
        response = sample_responses(model, tokenizer["instance"], row["prompt"], 1, cfg["max_new_tokens"], rng, device, precision)[0]
        text = tokenizer["instance"].decode(response, errors="replace")
        samples.append(dict(id=row["id"], prompt=row["prompt"], words=row["words"], text=text, new_tokens=len(response),
                            **reward(text, row["words"], cfg["min_words"])))
    synchronize(device)
    result = dict(signature=signature, model=name, tasks=len(samples),
                  success_rate=sum(s["success"] for s in samples) / len(samples),
                  mean_reward=sum(s["reward"] for s in samples) / len(samples),
                  keyword_coverage=sum(s["keyword_coverage"] for s in samples) / len(samples),
                  repetition=sum(s["repetition"] for s in samples) / len(samples),
                  mean_words=sum(s["word_count"] for s in samples) / len(samples),
                  generated_tokens=sum(s["new_tokens"] for s in samples), seconds=time.perf_counter() - began, samples=samples)
    write_json(result_path, result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def plot_alignment(root, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["model"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("success_rate", "keyword_coverage", "repetition"),
                                 ("Verified success", "Keyword coverage", "Word repetition proxy")):
        ax.bar(names, [r[metric] for r in results], color=["#9aa7ae", "#3a7b70", "#6598c4", "#d1a261"])
        ax.set(title=title, ylim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = root / "comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run_alignment(base_cfg, task):
    cfg = base_cfg["alignment"]
    validate_alignment(cfg)
    data_root, manifest, instance = prepare_alignment(cfg, base_cfg["progress"])
    tokenizer = dict(instance=instance, sha256=manifest["provenance"]["tokenizer_sha256"])
    root = Path(cfg["output_dir"]).resolve()
    if task == "align_prepare":
        return dict(prepared=True, output_dir=str(root), data=manifest)
    checkpoint = Path(cfg["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("未找到已有 0.1B 最佳权重，请检查 alignment.checkpoint")
    train_rows, dev_rows, test_rows = (read_examples(data_root / f"{split}.jsonl") for split in ("train", "dev", "test"))
    paths = dict(base=checkpoint)
    if task == "align":
        for method in ("sft", "dpo", "rlvr"):
            parent = checkpoint if method == "sft" else paths["sft"]
            paths[method], complete = train_stage(base_cfg, cfg, method, parent, manifest, tokenizer, train_rows, dev_rows)
            if not complete:
                return dict(completed=False, stage=method, message="已保存断点；续训将 alignment.stop_after 设为 null")
    else:
        paths.update({method: root / method / "final.pt" for method in ("sft", "dpo", "rlvr")})
    results = [evaluate_checkpoint(base_cfg, cfg, name, path, tokenizer, test_rows, manifest["signature"]) for name, path in paths.items()]
    training = {method: json.loads((root / method / "result.json").read_text(encoding="utf-8")) for method in ("sft", "dpo", "rlvr")}
    report = dict(completed=True, task="English keyword-conditioned story generation", data=manifest, evaluation=results,
                  training=training, checkpoints={name: str(path) for name, path in paths.items()},
                  limitations=["同一预训练基座，DPO/RLVR 都从同一 SFT 权重分支，冻结参考模型为该 SFT 权重。",
                               "自动约束偏好与可验证词语奖励，不等同于人工偏好或通用数学/对话对齐。",
                               "RLVR 为单次 on-policy 分组 REINFORCE + 逐 token 分布 KL，不是完整 PPO/GRPO 复现。",
                               "固定各方法配置的更新次数，训练和采样预算并不相等，仅一个训练种子。",
                               "关键词、长度和重复率仅是代理指标，需要人工阅读样例，不能保证故事情节质量。"])
    report["plot"] = plot_alignment(root, results)
    write_json(root / "report.json", report)
    fields = [key for key in results[0] if key not in ("signature", "samples")]
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    from src.report_tasks import export_report
    project_report = export_report(base_cfg)
    return dict(completed=True, output_dir=str(root), comparison=[{k:v for k,v in r.items() if k not in ("signature", "samples")} for r in results],
                report=str(root / "report.json"), plot=report["plot"], project_report=project_report["report"])

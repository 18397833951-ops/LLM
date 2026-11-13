"""从本地故事构造关键词条件任务；偏好是自动约束偏好，不是人工评分。"""

from collections import Counter
import itertools
import json
from pathlib import Path
import re
import tempfile

from src.data import file_hash, text_hash, write_json
from src.progress import progress_bar
from src.science_data import fingerprint, local_documents
from src.tokenizer import BPETokenizer


def task_prompt(words):
    return f"Write a short story that uses both words: {words[0]}, {words[1]}.\nStory:\n"


def pair_split(words):
    bucket = int(fingerprint(sorted(words))[:8], 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def reward(text, words, min_words):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    seen = set(tokens)
    coverage = sum(word.lower() in seen for word in words) / len(words)
    long_enough = len(tokens) >= min_words
    unique_ratio = len(seen) / max(1, len(tokens))
    non_repetitive = unique_ratio >= 0.4
    score = 0.75 * coverage + 0.15 * int(long_enough) + 0.10 * int(non_repetitive)
    return dict(reward=score, keyword_coverage=coverage, success=coverage == 1 and long_enough and non_repetitive,
                word_count=len(tokens), unique_word_ratio=unique_ratio, repetition=1 - unique_ratio)


def make_example(text, tokenizer, cfg, allowed_splits):
    # 按字符边界截断，再回编码，避免把单词截成不完整片段。
    snippet = tokenizer.decode(tokenizer.encode(text)[:cfg["max_new_tokens"] - 1], errors="replace")
    last_space = snippet.rfind(" ")
    if last_space > 0:
        snippet = snippet[:last_space]
    sentence_end = max(snippet.rfind(mark) for mark in ".!?")
    if sentence_end > 0 and len(re.findall(r"[A-Za-z]+", snippet[:sentence_end + 1])) >= cfg["min_words"]:
        snippet = snippet[:sentence_end + 1]
    present = [word for word in cfg["keywords"] if re.search(r"\b" + re.escape(word) + r"\b", snippet, re.I)]
    # 稳定地挑选一个分组合法的词对，不改变源故事内容来强行凑正例。
    pairs = sorted(itertools.combinations(present, 2), key=fingerprint)
    for words in pairs:
        split = pair_split(words)
        if split not in allowed_splits:
            continue
        prompt = task_prompt(words)
        if len(tokenizer.encode(prompt)) > cfg["max_prompt_tokens"] or not reward(snippet, words, cfg["min_words"])["success"]:
            continue
        replacement = next(word for word in cfg["keywords"] if word not in words)
        rejected = re.sub(r"\b" + re.escape(words[0]) + r"\b", replacement, snippet, flags=re.I)
        if reward(rejected, words, cfg["min_words"])["keyword_coverage"] >= 1:
            continue
        if len(tokenizer.encode(rejected, add_eos=True)) > cfg["max_new_tokens"]:
            continue
        return dict(prompt=prompt, words=list(words), chosen=snippet, rejected=rejected,
                    source_text_sha256=text_hash(text), split=split,
                    preference_source="synthetic: chosen contains both required words; rejected replaces one required word")
    return None


def prepare_alignment(cfg, progress_cfg):
    root = Path(cfg["source_dir"])
    parent = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sources = {split: root / f"{split}.jsonl" for split in ("train", "val")}
    for split, path in sources.items():
        if file_hash(path) != parent["splits"][split]["sha256"]:
            raise ValueError(f"alignment 的 {split} 来源校验失败")
    tokenizer_path = Path(cfg["tokenizer_dir"]) / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.metadata.get("train_file_sha256") != parent["splits"]["train"]["sha256"]:
        raise ValueError("alignment 词表来源与训练集不一致")
    provenance = dict(version=1, train_sha256=parent["splits"]["train"]["sha256"],
                      val_sha256=parent["splits"]["val"]["sha256"], tokenizer_sha256=file_hash(tokenizer_path),
                      options={key: cfg[key] for key in ("train_examples", "dev_examples", "test_examples", "max_scan",
                          "max_new_tokens", "max_prompt_tokens", "min_words", "keywords")})
    signature = fingerprint(provenance)
    output = Path(cfg["output_dir"]).resolve() / "data"
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("alignment 数据配置已改变，请使用新 alignment.output_dir")
        for split, info in manifest["files"].items():
            if file_hash(output / f"{split}.jsonl") != info["sha256"]:
                raise ValueError("alignment 数据文件被修改")
        return output, manifest, tokenizer
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = dict(train=cfg["train_examples"], dev=cfg["dev_examples"], test=cfg["test_examples"])
    rows = {name: [] for name in wanted}
    seen = set()
    stats = Counter()
    for source_split, destinations in (("train", {"train"}), ("val", {"dev", "test"})):
        with progress_bar(total=sum(wanted[s] for s in destinations), config=progress_cfg,
                          desc=f"Alignment {source_split}", unit="task") as bar:
            for index, text in enumerate(local_documents(sources[source_split])):
                if index >= cfg["max_scan"]:
                    break
                stats[f"scanned_{source_split}"] += 1
                example = make_example(text, tokenizer, cfg, {s for s in destinations if len(rows[s]) < wanted[s]})
                if example is None:
                    continue
                response_hash = text_hash(example["chosen"])
                if response_hash in seen:
                    continue
                seen.add(response_hash)
                example["id"] = f"{source_split}:{index}"
                rows[example["split"]].append(example)
                bar.update(1)
                if all(len(rows[s]) == wanted[s] for s in destinations):
                    break
    if any(len(rows[s]) != wanted[s] for s in wanted):
        raise ValueError(f"满足约束且按词对隔离的样本不足：{ {s:len(rows[s]) for s in rows} }；可减小 alignment 样本数")
    with tempfile.TemporaryDirectory(prefix=".alignment-data-", dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        files = {}
        for split, values in rows.items():
            path = staging / f"{split}.jsonl"
            path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
            files[split] = dict(examples=len(values), sha256=file_hash(path))
        manifest = dict(signature=signature, provenance=provenance, files=files, statistics=dict(stats),
                        split_rule="sorted keyword-pair SHA256 mod 10: 0=test, 1=dev, others=train",
                        limitations=["自动构造约束偏好，非人工偏好数据。", "验证和测试来自原始验证 split，词对不与后训练训练集重合。",
                                     "关键词和词数奖励不能保证语法、情节、事实或通用指令质量。"])
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    return output, manifest, tokenizer

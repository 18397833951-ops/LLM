"""TinyStories 数据准备与检查。核心文件处理只依赖标准库。"""

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import tempfile
import unicodedata
from datetime import datetime, timezone


def text_hash(text):
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_config(cfg):
    if cfg["train_split"] == cfg["val_split"]:
        raise ValueError("训练集与验证集必须使用不同的官方 split")
    for key in ("train_size", "val_size"):
        value = cfg[key]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{key} 必须为正整数或 null")
    for key in ("shuffle_buffer", "min_chars"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("shuffle", "deduplicate", "remove_overlap"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} 必须为 true 或 false")
    if type(cfg["seed"]) is not int:
        raise ValueError("seed 必须为整数")


def resolve_revision(cfg):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(cfg["name"], revision=cfg["revision"])
    if not info.sha:
        raise ValueError("无法解析数据集 commit，请检查网络或 revision")
    return info.sha


def load_split(cfg, revision, split):
    from datasets import load_dataset

    dataset = load_dataset(
        cfg["name"], name=cfg["subset"], revision=revision, split=split,
        streaming=True, cache_dir=str(Path(cfg["cache_dir"]).resolve()),
    )
    if cfg["shuffle"]:
        dataset = dataset.shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])
    return dataset


def save_split(rows, path, cfg, split, limit, blocked):
    """保留原文，只使用归一化指纹去重；不跨故事拼接。"""
    stats = {"scanned": 0, "saved": 0, "empty": 0, "too_short": 0,
             "duplicates_skipped": 0, "overlap_skipped": 0}
    seen = set()
    iterator = iter(rows)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        while limit is None or stats["saved"] < limit:
            try:
                row = next(iterator)
            except StopIteration:
                break
            stats["scanned"] += 1
            text = row.get(cfg["text_field"])
            if not isinstance(text, str):
                raise ValueError(f"{split} 第 {stats['scanned']} 条的文本字段缺失或非字符串")
            if not text.strip():
                stats["empty"] += 1
                continue
            if len(text.strip()) < cfg["min_chars"]:
                stats["too_short"] += 1
                continue
            fingerprint = text_hash(text)
            if cfg["deduplicate"] and fingerprint in seen:
                stats["duplicates_skipped"] += 1
                continue
            if cfg["remove_overlap"] and fingerprint in blocked:
                stats["overlap_skipped"] += 1
                continue
            item = {"id": f"{split}:{stats['scanned'] - 1}",
                    "text": text, "text_sha256": fingerprint}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            seen.add(fingerprint)
            stats["saved"] += 1
            if stats["saved"] % 1000 == 0:
                logging.info("%s: 已保存 %s 条", split, stats["saved"])

    if stats["saved"] == 0:
        raise ValueError(f"{split} 过滤后没有可用数据")
    if limit is not None and stats["saved"] < limit:
        raise ValueError(
            f"{split} 仅有 {stats['saved']} 条可用数据，少于目标 {limit}；"
            "请减小数量或设为 null，不会把不完整文件当作准备成功"
        )
    stats.update({"file": path.name, "sha256": file_hash(path),
                  "bytes": path.stat().st_size})
    return stats, seen


def prepare_data(cfg):
    validate_config(cfg)
    output = Path(cfg["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(
            f"输出目录已存在：{output}。请先 stat=inspect，或指定新的 data.output_dir；不自动覆盖。"
        )
    revision = resolve_revision(cfg)
    logging.info("数据源 %s，commit=%s", cfg["name"], revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 两组和 manifest 全部成功才发布目录，下载失败不会留下伪完整结果。
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        train_stats, train_hashes = save_split(
            load_split(cfg, revision, cfg["train_split"]), staging / "train.jsonl",
            cfg, cfg["train_split"], cfg["train_size"], set(),
        )
        val_stats, _ = save_split(
            load_split(cfg, revision, cfg["val_split"]), staging / "val.jsonl",
            cfg, cfg["val_split"], cfg["val_size"], train_hashes,
        )
        versions = {}
        for package in ("datasets", "huggingface-hub", "hydra-core"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": cfg["name"], "resolved_revision": revision,
            "config": cfg, "versions": versions,
            "sampling": "buffered shuffle" if cfg["shuffle"] else "source order",
            "fingerprint": "sha256(UTF-8(NFC + whitespace collapse))",
            "splits": {"train": train_stats, "val": val_stats},
        }
        write_json(staging / "manifest.json", manifest)
        # 不主动替换已存在目录，避免重跑覆盖之前的实验数据。
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return {"output_dir": str(output), **manifest}


def scan_file(path, sample_count, preview_chars):
    counts = {"rows": 0, "empty": 0, "duplicates": 0, "bad_fingerprints": 0}
    seen = set()
    samples = []
    total_chars = 0
    min_chars = None
    max_chars = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number} 不是合法 JSON") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path.name}:{line_number} 缺少字符串 text")
            text = row["text"]
            fingerprint = text_hash(text)
            counts["rows"] += 1
            counts["empty"] += int(not text.strip())
            counts["duplicates"] += int(fingerprint in seen)
            counts["bad_fingerprints"] += int(row.get("text_sha256") != fingerprint)
            seen.add(fingerprint)
            size = len(text)
            total_chars += size
            min_chars = size if min_chars is None else min(min_chars, size)
            max_chars = max(max_chars, size)
            if len(samples) < sample_count:
                samples.append({"id": row.get("id"), "text": text[:preview_chars]})
    counts.update({"min_chars": min_chars, "max_chars": max_chars,
                   "mean_chars": round(total_chars / counts["rows"], 2) if counts["rows"] else 0,
                   "sha256": file_hash(path), "samples": samples})
    return counts, seen


def inspect_data(cfg, options):
    for key in ("samples", "preview_chars"):
        if type(options[key]) is not int or options[key] < 0:
            raise ValueError(f"inspect.{key} 必须为非负整数")
    output = Path(cfg["output_dir"]).resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到 {manifest_path}，请先 python main.py stat=prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {"output_dir": str(output), "source": manifest["source"],
              "resolved_revision": manifest["resolved_revision"], "splits": {}, "issues": []}
    hashes = {}
    for split in ("train", "val"):
        stats, hashes[split] = scan_file(
            output / f"{split}.jsonl", options["samples"], options["preview_chars"]
        )
        expected = manifest["splits"][split]
        if stats["rows"] != expected["saved"]:
            report["issues"].append(f"{split}: 数量与 manifest 不一致")
        if stats["sha256"] != expected["sha256"]:
            report["issues"].append(f"{split}: 文件校验值与 manifest 不一致")
        for key in ("empty", "duplicates", "bad_fingerprints"):
            if stats[key]:
                report["issues"].append(f"{split}: {key}={stats[key]}")
        if not stats["rows"]:
            report["issues"].append(f"{split}: 没有数据")
        report["splits"][split] = stats
    report["overlap_unique_texts"] = len(hashes["train"] & hashes["val"])
    if report["overlap_unique_texts"]:
        report["issues"].append("训练集与验证集存在重复文本")
    report["passed"] = not report["issues"]
    write_json(output / "inspection.json", report)
    return report

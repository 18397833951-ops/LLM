"""Hydra 分词器任务，数据和词表均在本地，不访问网络。"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile

from src.data import file_hash, write_json
from src.tokenizer import BPETokenizer


def read_texts(path, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("文档数量必须为正整数或 null")
    with Path(path).open(encoding="utf-8") as stream:
        count = 0
        for line_number, line in enumerate(stream, 1):
            if limit is not None and count >= limit:
                break
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number} 缺少字符串 text")
            count += 1
            yield row["text"]
        if not count:
            raise ValueError(f"数据文件为空：{path}")


def data_manifest(data_cfg):
    root = Path(data_cfg["output_dir"]).resolve()
    return root, json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def verify_split(root, manifest, split):
    path = root / f"{split}.jsonl"
    if file_hash(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"{split} 文件校验失败，请先 stat=inspect")
    return path


def train_tokenizer(data_cfg, cfg):
    root, manifest = data_manifest(data_cfg)
    train_path = verify_split(root, manifest, "train")
    output = Path(cfg["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"词表目录已存在：{output}，请检查已有词表或修改 tokenizer.output_dir")
    tokenizer = BPETokenizer()
    logging.info("只使用 train.jsonl 学习 BPE，不读取验证集")
    metadata = tokenizer.train(
        read_texts(train_path, cfg["train_documents"]),
        cfg["vocab_size"], cfg["min_frequency"],
    )
    metadata.update({"data_source": manifest["source"],
                     "data_revision": manifest["resolved_revision"],
                     "train_file_sha256": file_hash(train_path),
                     "created_at_utc": datetime.now(timezone.utc).isoformat(),
                     "config": cfg})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=output.parent) as temporary:
        staging = Path(temporary) / "tokenizer"
        staging.mkdir()
        tokenizer.save(staging / "tokenizer.json")
        write_json(staging / "train_report.json", metadata)
        staging.rename(output)
    return {"output_dir": str(output), **metadata}


def check_tokenizer(data_cfg, cfg):
    root, manifest = data_manifest(data_cfg)
    output = Path(cfg["output_dir"]).resolve()
    tokenizer = BPETokenizer.load(output / "tokenizer.json")
    if type(cfg["preview_tokens"]) is not int or cfg["preview_tokens"] < 0:
        raise ValueError("tokenizer.preview_tokens 必须是非负整数")
    issues = []
    report = {"vocab_size": tokenizer.vocab_size, "splits": {}, "issues": issues}
    train_sha = verify_split(root, manifest, "train")
    if tokenizer.metadata.get("train_file_sha256") != file_hash(train_sha):
        raise ValueError("当前训练数据与词表训练来源不同，请检查路径")
    for split in ("train", "val"):
        path = verify_split(root, manifest, split)
        documents = total_bytes = total_chars = total_tokens = mismatches = 0
        for text in read_texts(path, cfg["check_documents"]):
            ids = tokenizer.encode(text)
            documents += 1
            total_bytes += len(text.encode("utf-8"))
            total_chars += len(text)
            total_tokens += len(ids)
            mismatches += int(tokenizer.decode(ids) != text)
        report["splits"][split] = {
            "documents": documents, "utf8_bytes": total_bytes,
            "characters": total_chars, "tokens": total_tokens,
            "bytes_per_token": round(total_bytes / total_tokens, 3) if total_tokens else None,
            "roundtrip_mismatches": mismatches,
        }
        if mismatches:
            issues.append(f"{split}: {mismatches} 篇文本无法准确还原")
    edge_cases = ["", " ", "\n\n\t", "Hello, world!", "中文也可以还原。",
                  "🙂 café e\u0301", "<eos> stays ordinary text", "a_b 12.34\r\n"]
    for text in edge_cases:
        if tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) != text:
            issues.append(f"特殊输入无法还原：{text!r}")
    report["edge_cases_checked"] = len(edge_cases)
    text = cfg["text"]
    ids = tokenizer.encode(text)
    # 再读一次文件验证持久化行为，不复用内存中的实例。
    reloaded = BPETokenizer.load(output / "tokenizer.json")
    if reloaded.encode(text) != ids:
        issues.append("保存加载后编码不一致")
    report["example"] = {"text": text, "token_count": len(ids),
                         "token_ids_preview": ids[:cfg["preview_tokens"]],
                         "decoded": tokenizer.decode(ids)}
    report["passed"] = not issues
    write_json(output / "check_report.json", report)
    return report

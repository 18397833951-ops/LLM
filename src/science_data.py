"""本地语料清洗、泄漏检查和共享词表编码，不联网。"""

from collections import Counter, defaultdict
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np

from src.data import file_hash, text_hash, write_json
from src.progress import progress_bar
from src.tokenizer import BPETokenizer


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local_documents(path, kind="jsonl"):
    """WET 按 Content-Length 读取，不把 WARC 头当作训练文本。"""
    path = Path(path)
    if kind == "jsonl":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"{path.name}:{number} 缺少字符串 text")
                yield row["text"]
    elif kind == "wet":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if not line.startswith(b"WARC/"):
                    raise ValueError("WET 记录缺少 WARC 头")
                headers = {}
                while True:
                    line = stream.readline()
                    if not line:
                        raise ValueError("WET 头意外结束")
                    if not line.strip():
                        break
                    key, separator, value = line.partition(b":")
                    if not separator:
                        raise ValueError("非法 WET 头")
                    headers[key.strip().lower()] = value.strip()
                size = int(headers.get(b"content-length", b"-1"))
                if not 0 <= size <= 64 * 1024 * 1024:
                    raise ValueError("WET Content-Length 缺失或超出 64MiB 限制")
                body = stream.read(size)
                if len(body) != size:
                    raise ValueError("WET 正文截断")
                if headers.get(b"warc-type") == b"conversion":
                    yield body.decode("utf-8", errors="replace")
    else:
        raise ValueError("source_format 只支持 jsonl/wet")


def clean_text(text, cfg):
    text = unicodedata.normalize("NFC", text)
    if cfg["strip_html"]:
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = text.strip()
    if not cfg["min_chars"] <= len(text) <= cfg["max_chars"]:
        return None, "length", 0
    letters = [c for c in text if c.isalpha()]
    latin = sum(c.isascii() for c in letters) / max(1, len(letters))
    if cfg["language_filter"] and latin < cfg["min_latin_ratio"]:
        return None, "language_heuristic", 0
    words = re.findall(r"[A-Za-z]+", text.lower())
    symbols = sum(not c.isalnum() and not c.isspace() for c in text) / max(1, len(text))
    if cfg["quality_filter"] and ("lorem ipsum" in text.lower() or not words or
            len(set(words)) / len(words) < cfg["min_unique_word_ratio"] or symbols > cfg["max_symbol_ratio"]):
        return None, "quality", 0
    redactions = 0
    if cfg["redact_pii"]:
        text, count = re.subn(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        redactions += count
        # 保守规则只覆盖长数字串/常见电话格式，不声称完整 PII 检测。
        def phone(match):
            return "[PHONE]" if sum(c.isdigit() for c in match.group()) >= 8 else match.group()
        old = text
        text = re.sub(r"\+?\d[\d ()-]{6,}\d", phone, text)
        redactions += text.count("[PHONE]") - old.count("[PHONE]")
    return text, None, redactions


class NearDuplicates:
    """MinHash + LSH 候选检索，候选最后用真实 shingle Jaccard 确认。"""
    def __init__(self, cfg):
        self.words = cfg["shingle_words"]
        self.threshold = cfg["jaccard_threshold"]
        self.permutations = cfg["permutations"]
        self.bands = cfg["bands"]
        self.rows = self.permutations // self.bands
        rng = np.random.default_rng(1729)
        self.a = rng.integers(1, 2 ** 63, size=self.permutations, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2 ** 63, size=self.permutations, dtype=np.uint64)
        self.buckets = defaultdict(list)
        self.documents = []

    def shingles(self, text):
        words = re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())
        if not words:
            words = [text]
        count = max(1, len(words) - self.words + 1)
        return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + self.words]).encode(), digest_size=8).digest(), "little")
                for i in range(count)}

    def keys(self, shingles):
        values = np.array(list(shingles), dtype=np.uint64)
        signature = (self.a[:, None] * values[None, :] + self.b[:, None]).min(axis=1)
        return [(band, signature[band * self.rows:(band + 1) * self.rows].tobytes())
                for band in range(self.bands)]

    def duplicate(self, text):
        shingles = self.shingles(text)
        candidates = set()
        for key in self.keys(shingles):
            candidates.update(self.buckets.get(key, ()))
        for index in sorted(candidates):
            other = self.shingles(self.documents[index])
            if len(shingles & other) / len(shingles | other) >= self.threshold:
                return True
        return False

    def add(self, text):
        index = len(self.documents)
        self.documents.append(text)
        for key in self.keys(self.shingles(text)):
            self.buckets[key].append(index)


def emit(stream, text, index):
    stream.write(json.dumps(dict(id=index, text=text, text_sha256=text_hash(text)), ensure_ascii=False) + "\n")


def prepare_corpora(cfg, progress_cfg):
    source_root = Path(cfg["source_dir"]).resolve()
    parent = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    train_path, val_path = (source_root / f"{split}.jsonl" for split in ("train", "val"))
    for split, path in (("train", train_path), ("val", val_path)):
        if file_hash(path) != parent["splits"][split]["sha256"]:
            raise ValueError(f"原始 {split} 文件与 manifest 不一致")
    source_path = Path(cfg["source_path"]).resolve() if cfg["source_path"] else train_path
    tokenizer_path = Path(cfg["tokenizer_dir"]) / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.metadata.get("train_file_sha256") != parent["splits"]["train"]["sha256"]:
        raise ValueError("共享词表不属于指定的原始训练集")
    provenance = dict(version=1, source_sha256=file_hash(source_path),
                      source_format=cfg["source_format"] if cfg["source_path"] else "jsonl",
                      parent_train_sha256=parent["splits"]["train"]["sha256"],
                      parent_val_sha256=parent["splits"]["val"]["sha256"],
                      train_documents=cfg["train_documents"], val_documents=cfg["val_documents"],
                      cleaning=cfg["cleaning"])
    key = fingerprint(provenance)
    output = Path(cfg["output_dir"]).resolve() / "data"
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["key"] != key:
            raise ValueError("实验数据配置已改变，请用新的 science.output_dir")
        for name, info in manifest["files"].items():
            if file_hash(output / f"{name}.jsonl") != info["sha256"]:
                raise ValueError("实验数据被修改，请检查输出目录")
        return output, manifest, tokenizer, file_hash(tokenizer_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_cfg = cfg["cleaning"]
    validation = list(local_documents(val_path))
    if len(validation) < cfg["val_documents"]:
        raise ValueError("验证故事数量不足")
    # 两组统一排除所有官方验证文本及 LSH 检出的近重复，验证集本身保持不变。
    blocked = {text_hash(text) for text in validation}
    val_index = NearDuplicates(clean_cfg)
    for text in progress_bar(validation, config=progress_cfg, desc="Index validation", unit="doc"):
        val_index.add(text)
    stats = Counter()
    seen = set()
    train_index = NearDuplicates(clean_cfg)
    with tempfile.TemporaryDirectory(prefix=".science-data-", dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        with (staging / "raw.jsonl").open("w", encoding="utf-8", newline="\n") as raw, \
             (staging / "clean.jsonl").open("w", encoding="utf-8", newline="\n") as clean:
            iterator = local_documents(source_path, provenance["source_format"])
            with progress_bar(total=cfg["train_documents"], config=progress_cfg, desc="Clean corpus", unit="doc") as bar:
                for text in iterator:
                    stats["scanned"] += 1
                    if not text.strip():
                        stats["empty"] += 1
                        continue
                    filtered, reason, redactions = clean_text(text, clean_cfg)
                    if (text_hash(text) in blocked or val_index.duplicate(text) or
                            (filtered is not None and filtered != text and (text_hash(filtered) in blocked or val_index.duplicate(filtered)))):
                        stats["validation_overlap"] += 1
                        continue
                    emit(raw, text, stats["raw"])
                    stats["raw"] += 1
                    bar.update(1)
                    if reason:
                        stats[reason] += 1
                    elif clean_cfg["exact_dedup"] and text_hash(filtered) in seen:
                        stats["exact_duplicate"] += 1
                    elif clean_cfg["near_dedup"] and train_index.duplicate(filtered):
                        stats["near_duplicate"] += 1
                    else:
                        emit(clean, filtered, stats["raw"] - 1)
                        stats["clean"] += 1
                        stats["redactions"] += redactions
                        seen.add(text_hash(filtered))
                        if clean_cfg["near_dedup"]:
                            train_index.add(filtered)
                    if stats["raw"] == cfg["train_documents"]:
                        break
        if stats["raw"] < cfg["train_documents"] or not stats["clean"]:
            raise ValueError("可用语料不足或清洗后为空，请调整 science 语料数量/过滤配置")
        with (staging / "val.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for index, text in enumerate(validation[:cfg["val_documents"]]):
                emit(stream, text, index)
        files = {name: dict(sha256=file_hash(staging / f"{name}.jsonl"),
                            documents=stats[name] if name != "val" else cfg["val_documents"])
                 for name in ("raw", "clean", "val")}
        manifest = dict(key=key, provenance=provenance, statistics=dict(stats), files=files,
                        source_description="local raw file" if cfg["source_path"] else "previously filtered TinyStories; additional-cleaning control")
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    return output, manifest, tokenizer, file_hash(tokenizer_path)


def encode_corpora(root, manifest, tokenizer, tokenizer_sha, output_root, context, progress_cfg):
    key = fingerprint(dict(data=manifest["key"], tokenizer=tokenizer_sha))
    output = Path(output_root) / "tokens"
    if output.exists():
        meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        if meta["key"] != key:
            raise ValueError("token 缓存来源不同，请使用新 science.output_dir")
        for name, info in meta["files"].items():
            if file_hash(output / f"{name}.bin") != info["sha256"]:
                raise ValueError("实验 token 缓存损坏")
    else:
        with tempfile.TemporaryDirectory(prefix=".science-tokens-", dir=output.parent) as temporary:
            staging = Path(temporary) / "tokens"
            staging.mkdir()
            dtype = "uint16" if tokenizer.vocab_size <= 65536 else "uint32"
            meta = dict(key=key, tokenizer_sha256=tokenizer_sha, dtype=dtype, files={})
            for name in ("raw", "clean", "val"):
                count = 0
                with (staging / f"{name}.bin").open("wb") as stream:
                    for text in progress_bar(local_documents(root / f"{name}.jsonl"), config=progress_cfg,
                                             total=manifest["files"][name]["documents"], desc=f"Encode {name}", unit="doc"):
                        ids = tokenizer.encode(text, add_eos=True)
                        stream.write(np.asarray(ids, dtype=dtype).tobytes())
                        count += len(ids)
                meta["files"][name] = dict(tokens=count, sha256=file_hash(staging / f"{name}.bin"))
            write_json(staging / "meta.json", meta)
            staging.rename(output)
    if any(info["tokens"] < context + 1 for info in meta["files"].values()):
        raise ValueError("实验语料不足一个上下文窗口")
    return output, meta

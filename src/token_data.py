"""把故事编码成含 EOS 的 token 流，使用磁盘缓存避免每次重编码。"""

import hashlib
import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.data import file_hash, write_json
from src.tokenizer import BPETokenizer
from src.tokenizer_tasks import data_manifest, read_texts, verify_split


def prepare_tokens(cfg):
    root, manifest = data_manifest(cfg["data"])
    tokenizer_path = Path(cfg["tokenizer"]["output_dir"]) / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_sha = file_hash(tokenizer_path)
    paths = {split: verify_split(root, manifest, split) for split in ("train", "val")}
    if tokenizer.metadata.get("train_file_sha256") != manifest["splits"]["train"]["sha256"]:
        raise ValueError("分词器与当前训练数据来源不同，请先检查路径")
    signature = {"version": 1, "tokenizer_sha256": tokenizer_sha,
                 "train_sha256": manifest["splits"]["train"]["sha256"],
                 "val_sha256": manifest["splits"]["val"]["sha256"]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output = Path(cfg["train"]["token_cache"]).resolve() / key
    if output.exists():
        meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        if meta["signature"] != signature:
            raise ValueError("token 缓存来源校验失败")
        for split in ("train", "val"):
            if file_hash(output / f"{split}.bin") != meta["splits"][split]["sha256"]:
                raise ValueError("token 缓存文件损坏，请更换 train.token_cache 路径")
        return tokenizer, output, meta
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = "uint16" if tokenizer.vocab_size <= 65536 else "uint32"
    meta = {"signature": signature, "key": key, "dtype": dtype, "splits": {}}
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=output.parent) as temporary:
        staging = Path(temporary) / "tokens"
        staging.mkdir()
        for split in ("train", "val"):
            count = documents = 0
            path = staging / f"{split}.bin"
            with path.open("wb") as stream:
                for text in read_texts(paths[split]):
                    ids = tokenizer.encode(text, add_eos=True)
                    stream.write(np.asarray(ids, dtype=dtype).tobytes())
                    count += len(ids)
                    documents += 1
            meta["splits"][split] = {"tokens": count, "documents": documents, "sha256": file_hash(path)}
            logging.info("%s 已编码 %s 篇，%s token（含 EOS）", split, documents, count)
        write_json(staging / "meta.json", meta)
        staging.rename(output)
    return tokenizer, output, meta


class TokenStream:
    def __init__(self, path, dtype, context):
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.context = context
        if len(self.tokens) < context + 1:
            raise ValueError("数据不足一个上下文窗口，请增加数据或减小 model.context")

    def sample(self, batch_size, generator, device):
        starts = torch.randint(len(self.tokens) - self.context, (batch_size,), generator=generator).tolist()
        # 拷贝只读 memmap 切片，避免 PyTorch 对不可写数组的警告。
        x = np.stack([self.tokens[start:start + self.context] for start in starts]).astype(np.int64)
        y = np.stack([self.tokens[start + 1:start + self.context + 1] for start in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def batches(self, batch_size, device, max_tokens=None):
        available = len(self.tokens) - 1
        if max_tokens is not None:
            if type(max_tokens) is not int or max_tokens < 1:
                raise ValueError("eval.max_tokens 必须是正整数或 null")
            available = min(available, max_tokens)
        start = 0
        while start < available:
            remaining = available - start
            length = min(self.context, remaining)
            batch = min(batch_size, remaining // length)
            count = length * batch
            x = np.array(self.tokens[start:start + count], dtype=np.int64).reshape(batch, length)
            y = np.array(self.tokens[start + 1:start + count + 1], dtype=np.int64).reshape(batch, length)
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            start += count

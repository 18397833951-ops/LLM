"""手写 byte-level BPE：统计相邻 token，按频次学习合并规则。"""

from collections import Counter, defaultdict
import heapq
import json
import logging
from pathlib import Path
import re


# 按字母、数字、空白、标点分块，所有字符都保留；训练和编码使用相同边界。
PATTERN = r"\s+|[^\W\d_]+|\d+|[^\w\s]+|_+"
SPLIT = re.compile(PATTERN, re.UNICODE)
SPECIALS = {"<pad>": 256, "<bos>": 257, "<eos>": 258}
FIRST_MERGE = 259


def merge_pair(tokens, pair, token_id):
    result = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            result.append(token_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {index: bytes([index]) for index in range(256)}
        self._cache = {}
        self.metadata = {}

    @property
    def vocab_size(self):
        return FIRST_MERGE + len(self.merges)

    @property
    def pad_id(self):
        return SPECIALS["<pad>"]

    @property
    def bos_id(self):
        return SPECIALS["<bos>"]

    @property
    def eos_id(self):
        return SPECIALS["<eos>"]

    def train(self, texts, vocab_size=8192, min_frequency=2):
        if type(vocab_size) is not int or vocab_size < FIRST_MERGE:
            raise ValueError(f"vocab_size 必须是至少 {FIRST_MERGE} 的整数")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency 必须是正整数")

        self.__init__()
        chunks = Counter()
        documents = total_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("训练样本必须为字符串")
            documents += 1
            total_bytes += len(text.encode("utf-8"))
            chunks.update(tuple(chunk.encode("utf-8")) for chunk in SPLIT.findall(text))
        if not chunks:
            raise ValueError("没有可训练的文本")

        pieces = list(chunks)
        weights = [chunks[piece] for piece in pieces]
        counts = Counter()
        locations = defaultdict(set)
        for index, piece in enumerate(pieces):
            for pair, count in Counter(zip(piece, piece[1:])).items():
                counts[pair] += count * weights[index]
                locations[pair].add(index)
        heap = [(-count, pair) for pair, count in counts.items()]
        heapq.heapify(heap)

        while self.vocab_size < vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            frequency = -negative_count
            # 堆中保留旧计数，用当前计数验证，避免昂贵的任意位置更新。
            if counts.get(pair, 0) != frequency:
                continue
            if frequency < min_frequency:
                break
            token_id = self.vocab_size
            self.merges[pair] = token_id
            self.vocab[token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            affected_pairs = set()
            for index in list(locations[pair]):
                old = pieces[index]
                new = merge_pair(old, pair, token_id)
                old_counts = Counter(zip(old, old[1:]))
                new_counts = Counter(zip(new, new[1:]))
                for changed_pair in old_counts.keys() | new_counts.keys():
                    change = new_counts[changed_pair] - old_counts[changed_pair]
                    if change:
                        counts[changed_pair] += change * weights[index]
                        affected_pairs.add(changed_pair)
                    if new_counts[changed_pair]:
                        locations[changed_pair].add(index)
                    else:
                        locations[changed_pair].discard(index)
                pieces[index] = new
            for changed_pair in affected_pairs:
                if counts[changed_pair] > 0:
                    heapq.heappush(heap, (-counts[changed_pair], changed_pair))
            if len(self.merges) % 500 == 0:
                logging.info("BPE 词表 %s/%s", self.vocab_size, vocab_size)

        self.metadata = {"train_documents": documents, "train_bytes": total_bytes,
                         "unique_chunks": len(chunks), "requested_vocab_size": vocab_size,
                         "actual_vocab_size": self.vocab_size, "min_frequency": min_frequency}
        return self.metadata

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        tokens = tuple(chunk.encode("utf-8"))
        while len(tokens) > 1:
            candidates = ((self.merges[pair], pair) for pair in zip(tokens, tokens[1:])
                          if pair in self.merges)
            best = min(candidates, default=None)
            if best is None:
                break
            token_id, pair = best
            tokens = merge_pair(tokens, pair, token_id)
        if len(self._cache) < 50000:
            self._cache[chunk] = tokens
        return tokens

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("encode 输入必须是字符串")
        ids = [self.bos_id] if add_bos else []
        for chunk in SPLIT.findall(text):
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens=True, errors="strict"):
        content = []
        special_names = {value: key for key, value in SPECIALS.items()}
        for token_id in ids:
            if type(token_id) is not int:
                raise ValueError("token ID 必须是整数")
            if token_id in special_names:
                if not skip_special_tokens:
                    content.append(special_names[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                content.append(self.vocab[token_id])
            else:
                raise ValueError(f"未知 token ID: {token_id}")
        return b"".join(content).decode("utf-8", errors=errors)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"分词器文件已存在：{path}")
        value = {"version": 1, "pattern": PATTERN, "special_tokens": SPECIALS,
                 "merges": [list(pair) for pair in self.merges], "metadata": self.metadata}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (value.get("version") != 1 or value.get("pattern") != PATTERN
                or value.get("special_tokens") != SPECIALS):
            raise ValueError("分词器格式、分块规则或特殊 token 不兼容")
        tokenizer = cls()
        if not isinstance(value.get("merges"), list):
            raise ValueError("缺少合并规则")
        for row in value["merges"]:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("合并规则格式错误")
            pair = tuple(row)
            if any(type(index) is not int or index not in tokenizer.vocab for index in pair):
                raise ValueError("合并引用了尚未定义的 token")
            if pair in tokenizer.merges:
                raise ValueError("存在重复合并规则")
            token_id = tokenizer.vocab_size
            tokenizer.merges[pair] = token_id
            tokenizer.vocab[token_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        tokenizer.metadata = value.get("metadata", {})
        return tokenizer

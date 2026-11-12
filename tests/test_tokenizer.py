from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.data import file_hash, write_json
from src.tokenizer import BPETokenizer, FIRST_MERGE, SPLIT, merge_pair
from src.tokenizer_tasks import check_tokenizer, train_tokenizer


def reference_train(texts, target):
    """小数据参考：每次重新统计全部片段，不使用增量索引或堆。"""
    sequences = [tuple(chunk.encode("utf-8")) for text in texts for chunk in SPLIT.findall(text)]
    rules = {}
    while FIRST_MERGE + len(rules) < target:
        counts = Counter(pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        if counts[pair] < 2:
            break
        index = FIRST_MERGE + len(rules)
        rules[pair] = index
        # 独立的非重叠替换逻辑，用来检验正式实现。
        replaced = []
        for sequence in sequences:
            output = []
            position = 0
            while position < len(sequence):
                if tuple(sequence[position:position + 2]) == pair:
                    output.append(index)
                    position += 2
                else:
                    output.append(sequence[position])
                    position += 1
            replaced.append(tuple(output))
        sequences = replaced
    return rules


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "work" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_incremental_counts_match_naive_reference(self):
        for seed in range(10):
            rng = random.Random(seed)
            texts = [" ".join(rng.choices(["abab", "aaaaa", "baba", "abcabc", "café", "你好"], k=20))
                     for _ in range(5)]
            tokenizer = BPETokenizer()
            tokenizer.train(texts, vocab_size=300)
            self.assertEqual(tokenizer.merges, reference_train(texts, 300))

    def test_bytes_fallback_preserves_unseen_unicode_and_whitespace(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["hello hello world"], 300)
        for text in ["", "  \t\r\n", "你好🙂", "e\u0301 é", "<eos>", "abc_123!\x00"]:
            with self.subTest(text=text):
                ids = tokenizer.encode(text)
                self.assertEqual(tokenizer.decode(ids), text)
                self.assertTrue(all(0 <= index < tokenizer.vocab_size for index in ids))

    def test_special_tokens_are_explicit(self):
        tokenizer = BPETokenizer()
        ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tokenizer.bos_id)
        self.assertEqual(ids[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode(ids), "hello")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), "<bos>hello<eos>")
        self.assertNotIn(tokenizer.eos_id, tokenizer.encode("<eos>"))

    def test_merge_does_not_cross_chunks_or_documents(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["a b", "a b", "c", "d"], 300)
        self.assertEqual(tokenizer.merges, {})

    def test_overlapping_pair_replacement(self):
        self.assertEqual(merge_pair((97, 97, 97, 97, 97), (97, 97), 259), (259, 259, 97))

    def test_persistence_and_overwrite_protection(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["abab abab abcabc abcabc"], 300)
        path = self.root / "tokenizer.json"
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.merges, tokenizer.merges)
        self.assertEqual(loaded.encode("abab abcabc 未见过🙂"), tokenizer.encode("abab abcabc 未见过🙂"))
        with self.assertRaises(FileExistsError):
            tokenizer.save(path)

    def test_invalid_vocabulary_is_rejected(self):
        value = {"version": 1, "pattern": SPLIT.pattern,
                 "special_tokens": {"<pad>": 256, "<bos>": 257, "<eos>": 258},
                 "merges": [[999, 0]]}
        path = self.root / "bad.json"
        write_json(path, value)
        with self.assertRaises(ValueError):
            BPETokenizer.load(path)

    def test_empty_training_and_invalid_ids(self):
        tokenizer = BPETokenizer()
        with self.assertRaises(ValueError):
            tokenizer.train([""], 300)
        for ids in [[True], [-1], [999]]:
            with self.assertRaises(ValueError):
                tokenizer.decode(ids)

    def test_train_and_check_local_files(self):
        dataset = self.root / "data"
        dataset.mkdir()
        splits = {"train": ["abab abab", "hello hello"], "val": ["unseen 中文🙂"]}
        manifest = {"source": "offline-fixture", "resolved_revision": "test", "splits": {}}
        for split, texts in splits.items():
            path = dataset / f"{split}.jsonl"
            path.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n"
                                    for text in texts), encoding="utf-8")
            manifest["splits"][split] = {"sha256": file_hash(path)}
        write_json(dataset / "manifest.json", manifest)
        data_cfg = {"output_dir": str(dataset)}
        cfg = {"output_dir": str(self.root / "tokenizer"), "train_documents": None,
               "check_documents": None, "vocab_size": 300, "min_frequency": 2,
               "text": "hello 中文🙂", "preview_tokens": 40}
        # 验证集即使损坏，训练也不读取它；只在检查任务读取。
        val = dataset / "val.jsonl"
        original = val.read_bytes()
        val.write_text("not JSON", encoding="utf-8")
        result = train_tokenizer(data_cfg, cfg)
        self.assertEqual(result["train_documents"], 2)
        val.write_bytes(original)
        report = check_tokenizer(data_cfg, cfg)
        self.assertTrue(report["passed"])
        self.assertEqual(report["splits"]["val"]["documents"], 1)
        with self.assertRaises(FileExistsError):
            train_tokenizer(data_cfg, cfg)
        with (dataset / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"text": "tampered"}\n')
        with self.assertRaises(ValueError):
            check_tokenizer(data_cfg, cfg)


if __name__ == "__main__":
    unittest.main()

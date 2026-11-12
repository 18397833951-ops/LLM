import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data import inspect_data, prepare_data, text_hash


class DataTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "work" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "prepared"
        self.cfg = dict(
            name="test/offline", revision="main", subset=None, text_field="text",
            train_split="train", val_split="validation", output_dir=str(self.output),
            cache_dir=str(Path(self.temporary.name) / "cache"), train_size=2, val_size=1,
            seed=42, shuffle=False, shuffle_buffer=100, min_chars=1,
            deduplicate=True, remove_overlap=True,
        )
        self.rows = {
            "train": [{"text": ""}, {"text": "Lily plays."}, {"text": "Lily  plays."},
                      {"text": "猫咪\nHello!"}],
            "validation": [{"text": "Lily plays."}, {"text": "A different story."}],
        }

    def prepare(self):
        with patch("src.data.resolve_revision", return_value="offline-test-commit"), \
             patch("src.data.load_split", side_effect=lambda cfg, rev, split: self.rows[split]):
            return prepare_data(self.cfg)

    def inspect(self):
        return inspect_data(self.cfg, {"samples": 2, "preview_chars": 100})

    def test_prepare_preserves_text_and_removes_duplicates_and_overlap(self):
        result = self.prepare()
        self.assertEqual(result["splits"]["train"]["empty"], 1)
        self.assertEqual(result["splits"]["train"]["duplicates_skipped"], 1)
        self.assertEqual(result["splits"]["val"]["overlap_skipped"], 1)
        rows = [json.loads(line) for line in (self.output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[1]["text"], "猫咪\nHello!")
        self.assertTrue(self.inspect()["passed"])

    def test_existing_output_is_not_overwritten(self):
        self.prepare()
        before = (self.output / "train.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, (self.output / "train.jsonl").read_bytes())

    def test_exhausted_source_does_not_publish_partial_dataset(self):
        self.cfg["val_size"] = 2
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_tampering_is_reported(self):
        self.prepare()
        with (self.output / "val.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"text": "Lily plays."}) + "\n")
        report = self.inspect()
        self.assertFalse(report["passed"])
        self.assertEqual(report["overlap_unique_texts"], 1)
        self.assertTrue(any("校验" in issue for issue in report["issues"]))

    def test_null_limit_reads_all(self):
        self.cfg["train_size"] = None
        self.assertEqual(self.prepare()["splits"]["train"]["saved"], 2)

    def test_empty_source_is_rejected(self):
        self.rows["train"] = [{"text": "  "}]
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_config_fails_before_network(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                self.cfg["train_size"] = value
                with self.assertRaises(ValueError):
                    prepare_data(self.cfg)

    def test_unicode_and_whitespace_fingerprint(self):
        self.assertEqual(text_hash("é  猫\n"), text_hash("e\u0301 猫"))


if __name__ == "__main__":
    unittest.main()

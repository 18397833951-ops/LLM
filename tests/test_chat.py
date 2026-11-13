import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_alignment import configuration
from src.chat_tasks import dialogue_prompt, fit_history, prepare_chat, run_chat
from src.data import file_hash
from src.model import GPT
from src.tokenizer import BPETokenizer
from src.web_app import DemoService


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for row in value:
            yield from strings(row)
    elif isinstance(value, dict):
        for row in value.values():
            yield from strings(row)


class ChatTests(unittest.TestCase):
    def test_history_retains_recent_complete_pairs(self):
        tokenizer = BPETokenizer()
        history = [dict(role="user", content="Old message " * 4), dict(role="assistant", content="Old reply " * 4),
                   dict(role="user", content="Hi"), dict(role="assistant", content="Hello!")]
        prompt, count = fit_history(tokenizer, 100, history, "How are you?", 20)
        self.assertEqual(count, 1)
        self.assertNotIn("Old message", prompt)
        self.assertIn("User: Hi\nAssistant: Hello!", prompt)
        self.assertTrue(prompt.endswith("User: How are you?\nAssistant:"))
        with self.assertRaises(ValueError):
            fit_history(tokenizer, 100, [dict(role="system", content="bad")], "hello", 20)
        with self.assertRaises(ValueError):
            fit_history(tokenizer, 100, [], "word " * 100, 20)
        self.assertEqual(dialogue_prompt([], "Hello!"), "User: Hello!\nAssistant:")

    def test_greeting_sft_and_web_chat(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir="work") as directory:
            root = Path(directory).resolve()
            tokenizer = BPETokenizer()
            seed_data = json.loads(Path("resources/english_dialogue.json").read_text(encoding="utf-8"))
            tokenizer.train(strings(seed_data), 1000, 2)
            token_root = root / "tokenizer"
            token_root.mkdir()
            tokenizer.save(token_root / "tokenizer.json")
            model = GPT(tokenizer.vocab_size, 1, 16, 2, 256)
            checkpoint = root / "base.pt"
            torch.save(dict(version=1, model=model.state_dict(), model_config=dict(vocab_size=tokenizer.vocab_size,
                       layers=1, width=16, heads=2, context=256, dropout=0.1, ffn_mult=4),
                       tokenizer_sha256=file_hash(token_root / "tokenizer.json")), checkpoint)
            cfg = configuration()
            cfg["train"].update(device="cpu", precision="fp32")
            cfg["progress"]["enabled"] = False
            cfg["chat"].update(checkpoint=str(checkpoint), tokenizer_dir=str(token_root), output_dir=str(root / "chat"),
                                batch_size=2, grad_accum=1, eval_every=1, save_every=1, stop_after=1)
            cfg["chat"]["sft"].update(steps=2, warmup_steps=0)
            prepared = run_chat(cfg, "chat_prepare")
            self.assertEqual(prepared["dataset"]["files"]["train"]["examples"], 400)
            self.assertEqual(prepared["dataset"]["files"]["test"]["examples"], 36)
            data_root = root / "chat" / "data"
            prompts = {split:{row["prompt"] for row in map(json.loads, (data_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())}
                       for split in ("train", "dev", "test")}
            self.assertFalse(prompts["train"] & prompts["test"])
            self.assertFalse(run_chat(cfg, "chat")["completed"])
            cfg["chat"]["stop_after"] = None
            result = run_chat(cfg, "chat")
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["evaluation"]), 2)
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report["evaluation"][1]["samples"]), 36)
            service = DemoService(cfg)
            self.assertTrue(service.summary()["models"]["chat"])
            response = service.generate(dict(model="chat", mode="chat", prompt="Hello!", history=[], max_new_tokens=3, seed=0))
            self.assertEqual(response["history_turns_used"], 0)
            self.assertLessEqual(response["new_tokens"], 3)
            history = [dict(role="user", content="Hello!"), dict(role="assistant", content="Hello!")]
            response = service.generate(dict(model="chat", mode="chat", prompt="How are you?", history=history, max_new_tokens=3, seed=0))
            self.assertEqual(response["history_turns_used"], 1)
            with self.assertRaises(ValueError):
                service.generate(dict(model="base", mode="chat", prompt="Hello!"))

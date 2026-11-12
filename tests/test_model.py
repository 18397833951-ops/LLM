from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from src.data import file_hash, write_json
from src.model import CausalAttention, GPT
from src.token_data import TokenStream
from src.tokenizer import BPETokenizer
from src.training import benchmark_or_train, generate, load_checkpoint, run_task, setup, training_plan, processed_tokens, update


torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_attention_matches_loop_reference(self):
        torch.manual_seed(0)
        attention = CausalAttention(8, 2, 4, 0).eval()
        x = torch.randn(1, 4, 8)
        q, k, v = attention.qkv(x).chunk(3, dim=-1)
        reference = torch.zeros_like(x)
        for position in range(4):
            for head in range(2):
                section = slice(head * 4, (head + 1) * 4)
                scores = q[0, position, section] @ k[0, :position + 1, section].T / 2
                reference[0, position, section] = scores.softmax(-1) @ v[0, :position + 1, section]
        torch.testing.assert_close(attention(x), attention.proj(reference))

    def test_future_tokens_do_not_change_earlier_logits(self):
        model = GPT(32, 2, 16, 2, 8, dropout=0).eval()
        ids = torch.randint(0, 32, (2, 8))
        modified = ids.clone()
        modified[:, 4:] = (modified[:, 4:] + 1) % 32
        torch.testing.assert_close(model(ids)[0][:, :4], model(modified)[0][:, :4])

    def test_loss_and_gradients_and_tied_weights(self):
        model = GPT(32, 1, 16, 2, 8, dropout=0)
        ids = torch.randint(0, 32, (2, 9))
        logits, loss = model(ids[:, :-1], ids[:, 1:])
        expected = F.cross_entropy(logits.reshape(-1, 32), ids[:, 1:].reshape(-1))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIs(model.output.weight, model.token_embedding.weight)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_can_overfit_one_batch(self):
        torch.manual_seed(7)
        model = GPT(16, 1, 16, 2, 8, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        y = x + 1
        initial = float(model(x, y)[1])
        for _ in range(60):
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
        self.assertLess(float(model(x, y)[1]), initial * 0.1)

    def test_default_parameter_count(self):
        with torch.device("meta"):
            model = GPT(8192, 14, 768, 12, 256)
        self.assertEqual(model.parameter_count(), 105719808)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        scratch = root / "work" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        data.mkdir()
        manifest = {"source": "offline-test", "resolved_revision": "test", "splits": {}}
        for split, texts in {
            "train": ["The little cat played with a red ball. " * 4] * 4,
            "val": ["A small bird visited the garden. " * 4],
        }.items():
            import json
            path = data / f"{split}.jsonl"
            path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
            manifest["splits"][split] = {"sha256": file_hash(path)}
        write_json(data / "manifest.json", manifest)
        token_dir = self.root / "tokenizer"
        token_dir.mkdir()
        tokenizer = BPETokenizer()
        tokenizer.train(["The little cat played with a red ball. " * 4], 280)
        tokenizer.metadata["train_file_sha256"] = manifest["splits"]["train"]["sha256"]
        tokenizer.save(token_dir / "tokenizer.json")
        self.tokenizer = tokenizer
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            cfg = compose(config_name="config", overrides=[
                "model.layers=1", "model.width=16", "model.heads=2", "model.context=8",
                "train.device=cpu", "train.precision=fp32", "train.max_steps=6", "train.warmup_steps=1",
                "train.batch_size=2", "train.grad_accum=2", "train.eval_every=3", "train.sample_every=3",
                "train.save_every=3", "train.log_every=1", "generate.max_new_tokens=3", "eval.max_tokens=17",
                "benchmark.warmup_steps=1", "benchmark.steps=2",
            ])
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        self.cfg["progress"]["enabled"] = False
        self.cfg["data"]["output_dir"] = str(data)
        self.cfg["tokenizer"]["output_dir"] = str(token_dir)
        self.cfg["train"]["token_cache"] = str(self.root / "tokens")
        self.cfg["train"]["output_dir"] = str(self.root / "train")
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()

    def test_memmap_shift_and_evaluation_covers_tail(self):
        path = self.root / "test.bin"
        np.arange(26, dtype="uint16").tofile(path)
        stream = TokenStream(path, "uint16", 8)
        x, y = stream.sample(2, torch.Generator().manual_seed(1), torch.device("cpu"))
        torch.testing.assert_close(y, x + 1)
        batches = list(stream.batches(2, torch.device("cpu")))
        self.assertEqual(sum(y.numel() for x, y in batches), 25)
        self.assertEqual(int(batches[-1][1][0, -1]), 25)

    def test_epoch_covers_every_target_once_including_tail(self):
        path = self.root / "epoch.bin"
        np.arange(70, dtype="uint16").tofile(path)
        stream = TokenStream(path, "uint16", 8)
        cfg = {**self.cfg["train"], "batch_size": 2, "grad_accum": 3,
               "epochs": 3, "max_steps": None}
        plan = training_plan(cfg, stream)
        self.assertEqual(plan["steps_per_epoch"], 2)
        self.assertEqual(plan["total_steps"], 6)
        for _ in range(3):
            targets = []
            for index in range(2):
                for x, y in stream.epoch_batches(index, 2, 3, torch.device("cpu")):
                    targets.extend(y[y != -100].tolist())
            self.assertEqual(targets, list(range(1, 70)))
        self.assertEqual(processed_tokens(1, plan), 48)
        self.assertEqual(processed_tokens(2, plan), 69)
        self.assertEqual(processed_tokens(6, plan), 207)

    def test_epoch_tail_gradient_uses_valid_token_weights(self):
        import copy
        path = self.root / "tail.bin"
        np.arange(70, dtype="uint16").tofile(path)
        stream = TokenStream(path, "uint16", 8)
        model = GPT(128, 1, 16, 2, 8, dropout=0)
        reference = copy.deepcopy(model)
        cfg = {**self.cfg["train"], "batch_size": 2, "grad_accum": 3, "grad_clip": 1e9}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        batches = list(stream.epoch_batches(1, 2, 3, torch.device("cpu")))
        x, y = [torch.cat(parts) for parts in zip(*batches)]
        _, loss = reference(x, y)
        loss.backward()
        reference_optimizer.step()
        actual = update(model, stream, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                        torch.Generator(), cfg, torch.device("cpu"), "fp32", epoch_step=1)
        self.assertAlmostEqual(actual, float(loss.detach()), places=5)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)

    def test_epoch_resume_and_epoch_end_validation(self):
        import copy
        import json
        full = copy.deepcopy(self.cfg)
        full["train"].update(max_steps=None, epochs=2, eval_every=5000, save_every=5000,
                             sample_every=5000, output_dir=str(self.root / "full_epochs"))
        result = benchmark_or_train(full, "train", self.run_dir)
        reference = load_checkpoint(Path(full["train"]["output_dir"]) / "last.pt")
        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(reference["seen_tokens"], result["training_plan"]["targets_per_epoch"] * 2)
        rows = [json.loads(line) for line in (Path(full["train"]["output_dir"]) / "metrics.jsonl").read_text().splitlines()]
        boundaries = [row for row in rows if row.get("epoch_end")]
        self.assertEqual(len(boundaries), 2)
        self.assertTrue(all("val_loss" in row for row in boundaries))
        resumed_cfg = copy.deepcopy(full)
        resumed_cfg["train"].update(output_dir=str(self.root / "resumed_epochs"), stop_after=1)
        benchmark_or_train(resumed_cfg, "train", self.run_dir)
        checkpoint = Path(resumed_cfg["train"]["output_dir"]) / "last.pt"
        resumed_cfg["train"].update(resume=str(checkpoint), stop_after=None)
        benchmark_or_train(resumed_cfg, "train", self.run_dir)
        resumed = load_checkpoint(checkpoint)
        for key in reference["model"]:
            torch.testing.assert_close(reference["model"][key], resumed["model"][key], rtol=0, atol=0)

    def test_keyboard_interrupt_keeps_last_periodic_checkpoint(self):
        import src.training as training
        self.cfg["train"].update(max_steps=None, epochs=2, save_every=2, eval_every=5000, sample_every=5000)
        original_update = training.update
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt
            return original_update(*args, **kwargs)
        with patch("src.training.update", side_effect=interrupt) as mocked_update:
            with self.assertRaises(KeyboardInterrupt):
                benchmark_or_train(self.cfg, "train", self.run_dir)
        # Mock 的调用记录会持有 memmap 参数，释放后 Windows 才允许清理测试目录。
        mocked_update.reset_mock()
        checkpoint = load_checkpoint(Path(self.cfg["train"]["output_dir"]) / "last.pt")
        self.assertEqual(checkpoint["step"], 2)

    def test_resume_matches_uninterrupted_training(self):
        import copy
        full = copy.deepcopy(self.cfg)
        full["train"]["output_dir"] = str(self.root / "full")
        benchmark_or_train(full, "train", self.run_dir)
        reference = load_checkpoint(Path(full["train"]["output_dir"]) / "last.pt")
        self.cfg["train"]["stop_after"] = 3
        benchmark_or_train(self.cfg, "train", self.run_dir)
        checkpoint_path = Path(self.cfg["train"]["output_dir"]) / "last.pt"
        self.cfg["train"]["resume"] = str(checkpoint_path)
        self.cfg["train"]["stop_after"] = None
        result = benchmark_or_train(self.cfg, "train", self.run_dir)
        resumed = load_checkpoint(checkpoint_path)
        self.assertTrue(result["completed_plan"])
        self.assertEqual(resumed["seen_tokens"], 192)
        for key in reference["model"]:
            torch.testing.assert_close(reference["model"][key], resumed["model"][key], rtol=0, atol=0)
        self.assertEqual(reference["best_val_loss"], resumed["best_val_loss"])

    def test_benchmark_eval_generation_and_plot(self):
        result = benchmark_or_train(self.cfg, "benchmark", self.run_dir)
        self.assertGreater(result["tokens_per_second"], 0)
        self.assertFalse(Path(self.cfg["train"]["output_dir"]).exists())
        benchmark_or_train(self.cfg, "train", self.run_dir)
        path = str(Path(self.cfg["train"]["output_dir"]) / "best.pt")
        self.cfg["eval"]["checkpoint"] = path
        self.cfg["generate"]["checkpoint"] = path
        evaluation = run_task(self.cfg, "eval", str(self.run_dir))
        self.assertEqual(evaluation["tokens"], 17)
        generated = run_task(self.cfg, "generate", str(self.run_dir))
        self.assertTrue(generated["text"].startswith(self.cfg["generate"]["prompt"]))
        plot = run_task(self.cfg, "plot", str(self.run_dir))
        self.assertGreater(Path(plot["curve"]).stat().st_size, 100)

    def test_generation_preserves_training_rng(self):
        model = GPT(self.tokenizer.vocab_size, 1, 16, 2, 8)
        state = torch.get_rng_state()
        generate(model, self.tokenizer, self.cfg["generate"], torch.device("cpu"), "fp32")
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)

    def test_cuda_device_without_index_is_normalized(self):
        self.cfg["train"]["device"] = "cuda"
        self.cfg["train"]["precision"] = "auto"
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True), \
             patch("torch.cuda.set_device") as select:
            device, precision = setup(self.cfg)
        self.assertEqual(str(device), "cuda:0")
        self.assertEqual(precision, "bf16")
        select.assert_called_once_with(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()

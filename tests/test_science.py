import copy
import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from src.data import file_hash, write_json
from src.science_data import clean_text, local_documents, NearDuplicates
from src.science_fit import fit_power_law, predict, summarize_scaling
from src.science_tasks import experiment_runs, run_science, validate_science
from src.tokenizer import BPETokenizer


def configuration():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return OmegaConf.to_container(compose(config_name="config"), resolve=True)


class ScienceTests(unittest.TestCase):
    def test_cleaning_switches_and_near_duplicate_confirmation(self):
        cfg = configuration()["science"]["cleaning"]
        cfg["min_chars"] = 10
        text = "<p>A little cat played near the forest and shared a toy with a happy dog. Contact cat@example.com or 13812345678.</p>"
        clean, reason, redactions = clean_text(text, cfg)
        self.assertIsNone(reason)
        self.assertNotIn("<p>", clean)
        self.assertNotIn("cat@example.com", clean)
        self.assertNotIn("13812345678", clean)
        self.assertEqual(redactions, 2)
        self.assertEqual(clean_text("中文故事讲述了小猫和小狗在森林里面玩耍。", cfg)[1], "language_heuristic")
        self.assertEqual(clean_text("lorem ipsum is a placeholder for a story.", cfg)[1], "quality")
        cfg.update(strip_html=False, redact_pii=False)
        self.assertEqual(clean_text(text, cfg)[0], text)
        index = NearDuplicates(cfg)
        story = " ".join(f"word{i}" for i in range(100))
        index.add(story)
        self.assertTrue(index.duplicate(story))
        self.assertTrue(index.duplicate(story + " ending"))
        self.assertFalse(index.duplicate("Completely unrelated text about a dog and the moon."))

    def test_wet_parser_uses_record_lengths_and_checks_truncation(self):
        with tempfile.TemporaryDirectory(dir="work") as directory:
            path = Path(directory) / "sample.wet"
            body = "A short story.\nWARC/ is part of the body.".encode()
            path.write_bytes(b"WARC/1.0\r\nWARC-Type: conversion\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body + b"\r\n\r\n")
            self.assertEqual(list(local_documents(path, "wet")), [body.decode()])
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaisesRegex(ValueError, "截断"):
                list(local_documents(path, "wet"))

    def test_power_law_prediction_and_holdout_is_not_fitted(self):
        sizes = np.array([2e6, 8e6, 14e6, 30e6])
        losses = 0.5 + 2 * (sizes / 1e6) ** -0.4
        options = dict(alpha_min=0.1, alpha_max=1, grid_points=301, bootstrap_samples=10)
        fit = fit_power_law(sizes, losses, options)
        self.assertAlmostEqual(float(predict(fit, 100e6)), 0.5 + 2 * 100 ** -0.4, delta=0.01)
        rows = [dict(model=f"m{i}", corpus="clean", seed=seed, parameters=int(n), final_val_loss=float(loss))
                for i, (n, loss) in enumerate(zip(sizes, losses)) for seed in (1, 2)]
        rows.extend(dict(model="target", corpus="clean", seed=seed, parameters=100000000, final_val_loss=5.0) for seed in (1, 2))
        result = summarize_scaling(rows, [f"m{i}" for i in range(4)], "target", options, 0)
        self.assertEqual(result["holdout"]["actual_loss"], 5.0)
        self.assertLess(result["holdout"]["predicted_loss"], 1)
        self.assertGreater(result["holdout"]["absolute_error"], 4)
        bad = copy.deepcopy(rows)
        bad.pop(0)
        with self.assertRaises(ValueError):
            summarize_scaling(bad, [f"m{i}" for i in range(4)], "target", options, 0)

    def test_config_budgets_and_run_pairing(self):
        cfg = configuration()["science"]
        validate_science(cfg)
        runs = experiment_runs(cfg)
        self.assertEqual(len(runs), 18)
        self.assertEqual(sum(r["corpus"] == "raw" for r in runs), 3)
        cfg["token_budget"] += 1
        with self.assertRaisesRegex(ValueError, "整数倍"):
            validate_science(cfg)

    def test_complete_cpu_pipeline_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir="work") as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            train = [f"The cat {i} found a bright red toy and played in the green garden with a little dog." for i in range(12)]
            train.append(train[0])
            train.append("<p>The brave rabbit shared a snack with an elephant beside a very tall mountain.</p>")
            validation = ["A blue spaceship travelled beyond the stars and returned home to its crew.",
                          "During winter a wise owl taught her friends how to build a warm shelter."]
            train.insert(0, validation[0])
            for name, texts in (("train", train), ("val", validation)):
                (source / f"{name}.jsonl").write_text("".join(json.dumps(dict(text=text)) + "\n" for text in texts), encoding="utf-8")
            write_json(source / "manifest.json", dict(source="test", resolved_revision="test", splits={
                split: dict(sha256=file_hash(source / f"{split}.jsonl")) for split in ("train", "val")}))
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            tokenizer = BPETokenizer()
            tokenizer.train([text for text in train if text not in validation], 280, 2)
            tokenizer.metadata["train_file_sha256"] = file_hash(source / "train.jsonl")
            tokenizer.save(tokenizer_dir / "tokenizer.json")
            cfg = configuration()
            cfg["train"].update(device="cpu", precision="fp32")
            cfg["progress"]["enabled"] = False
            science = cfg["science"]
            science.update(source_dir=str(source), tokenizer_dir=str(tokenizer_dir),
                           output_dir=str(root / "experiment"), train_documents=14, val_documents=2,
                           seeds=[9], token_budget=96, batch_size=1, grad_accum=1, context=8,
                           warmup_steps=1, eval_every=4, save_every=4, stop_after=4)
            science["cleaning"].update(min_chars=10, near_dedup=False)
            science["models"] = [dict(name=f"tiny{i}", layers=1, width=width, heads=2)
                                 for i, width in enumerate((8, 12, 16, 20))]
            science["comparison_model"] = "tiny2"
            science["holdout_model"] = dict(name="held", layers=1, width=24, heads=2)
            science["fit"].update(grid_points=30, bootstrap_samples=3)
            prepared = run_science(cfg, "science_prepare")
            self.assertEqual(prepared["statistics"]["exact_duplicate"], 1)
            self.assertEqual(prepared["statistics"]["validation_overlap"], 1)
            partial = run_science(cfg, "science")
            self.assertFalse(partial["completed"])
            self.assertEqual(partial["step"], 4)
            science["stop_after"] = None
            result = run_science(cfg, "science")
            self.assertTrue(result["completed"])
            self.assertEqual(result["experiments"], 6)
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            data_dir = root / "experiment" / "data"
            self.assertNotIn(validation[0], list(local_documents(data_dir / "raw.jsonl")))
            self.assertEqual(len(report["plots"]), 3)
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in report["plots"]))
            checkpoint = root / "experiment" / "models" / "tiny2_raw_s9" / "last.pt"
            resumed = torch.load(checkpoint, weights_only=True)
            cfg2 = copy.deepcopy(cfg)
            cfg2["science"]["output_dir"] = str(root / "continuous")
            run_science(cfg2, "science")
            continuous = torch.load(root / "continuous" / "models" / "tiny2_raw_s9" / "last.pt", weights_only=True)
            for key in resumed["model"]:
                torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
            self.assertTrue(run_science(cfg, "science_analyze")["completed"])
            science["lr"] *= 2
            with self.assertRaisesRegex(ValueError, "超参数"):
                run_science(cfg, "science_analyze")

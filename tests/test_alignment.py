import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.alignment_data import pair_split, reward, task_prompt
from src.alignment_math import dpo_loss, reinforce_loss, response_batch, response_log_probs
from src.alignment_tasks import run_alignment, sample_responses, train_stage, read_examples
from src.data import file_hash, write_json
from src.model import GPT
from src.tokenizer import BPETokenizer


def configuration():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return OmegaConf.to_container(compose(config_name="config"), resolve=True)


class AlignmentTests(unittest.TestCase):
    def test_response_mask_predicts_only_response_and_eos(self):
        tokenizer = BPETokenizer()
        examples = [dict(prompt="Ask:\n", chosen="cat"), dict(prompt="Q:\n", chosen="a")]
        x, targets = response_batch(examples, tokenizer, 32, torch.device("cpu"))
        for index, row in enumerate(examples):
            begin = len(tokenizer.encode(row["prompt"])) - 1
            self.assertTrue((targets[index, :begin] == -100).all())
            expected = tokenizer.encode(row["chosen"], add_eos=True)
            self.assertEqual(targets[index][targets[index] != -100].tolist(), expected)
        logits = torch.zeros(*x.shape, tokenizer.vocab_size, requires_grad=True)
        scores, _, mask = response_log_probs(logits, targets)
        torch.testing.assert_close(scores, -mask.sum(-1) * math.log(tokenizer.vocab_size))
        (-scores.mean()).backward()
        self.assertTrue((logits.grad[targets == -100] == 0).all())

    def test_dpo_formula_and_policy_gradient_direction(self):
        chosen = torch.tensor([1.0], requires_grad=True)
        rejected = torch.tensor([1.0], requires_grad=True)
        loss, margin = dpo_loss(chosen, rejected, torch.tensor([0.2]), torch.tensor([0.2]), 0.1)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        loss.backward()
        self.assertLess(float(chosen.grad), 0)
        self.assertGreater(float(rejected.grad), 0)
        policy = torch.zeros(2, 2, 6, requires_grad=True)
        targets = torch.tensor([[1, 2], [3, -100]])
        value, stats = reinforce_loss(policy, policy.detach().clone(), targets, torch.tensor([1.0, 0.0]), 0.02, (0, 5))
        value.backward()
        self.assertLess(float(policy.grad[0, 0, 1]), 0)
        self.assertGreater(float(policy.grad[1, 0, 3]), 0)
        self.assertAlmostEqual(stats["kl"], 0)
        self.assertTrue((policy.grad[:, :, [0, 5]] == 0).all())
        value, stats = reinforce_loss(policy.detach().requires_grad_(), policy.detach(), targets, torch.ones(2), 0.02, (0, 5))
        self.assertTrue(stats["zero_advantage_group"])
        self.assertAlmostEqual(float(value), 0)

    def test_reward_uses_word_boundaries_and_not_prompt(self):
        self.assertEqual(reward("A cat and a tree played in a garden together.", ["cat", "tree"], 5)["success"], True)
        self.assertEqual(reward("A cathedral stands on the street.", ["cat", "tree"], 5)["keyword_coverage"], 0)
        self.assertFalse(reward("cat tree cat tree cat tree", ["cat", "tree"], 6)["success"])
        self.assertEqual(pair_split(["cat", "tree"]), pair_split(["tree", "cat"]))

    def test_sampler_does_not_add_unobserved_eos_and_is_repeatable(self):
        class AlwaysOne(torch.nn.Module):
            context = 100
            def forward(self, x):
                logits = torch.full((*x.shape, 259), -1000.0)
                logits[:, :, 65] = 1000
                return logits, None
        model = AlwaysOne()
        tokenizer = BPETokenizer()
        def sample():
            return sample_responses(model, tokenizer, "x", 2, 3, torch.Generator().manual_seed(1), torch.device("cpu"), "fp32")
        self.assertEqual(sample(), [[65, 65, 65], [65, 65, 65]])
        self.assertEqual(sample(), sample())

    def test_full_pipeline_cpu_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir="work") as directory:
            root = Path(directory).resolve()
            source = root / "source"
            source.mkdir()
            vocabulary = ["cat", "dog", "tree", "river", "bird", "moon", "rabbit", "forest"]
            texts = {"train": [], "val": []}
            for a, b in itertools.combinations(vocabulary, 2):
                texts["train"].append(f"Once upon a time, a {a} and a {b} played together. They found a bright toy and shared it happily with their good friends.")
                texts["val"].append(f"Yesterday, a {a} and a {b} found a nice place to play together. Everyone shared their toys and had a wonderful time outside.")
            for split, values in texts.items():
                (source / f"{split}.jsonl").write_text("".join(json.dumps(dict(text=text)) + "\n" for text in values), encoding="utf-8")
            write_json(source / "manifest.json", dict(splits={s:dict(sha256=file_hash(source / f"{s}.jsonl")) for s in texts}))
            token_root = root / "tokenizer"
            token_root.mkdir()
            tokenizer = BPETokenizer()
            tokenizer.train(texts["train"], 400, 2)
            tokenizer.metadata["train_file_sha256"] = file_hash(source / "train.jsonl")
            tokenizer.save(token_root / "tokenizer.json")
            model = GPT(tokenizer.vocab_size, 1, 16, 2, 160)
            checkpoint = root / "base.pt"
            torch.save(dict(version=1, model=model.state_dict(), model_config=dict(vocab_size=tokenizer.vocab_size,
                       layers=1, width=16, heads=2, context=160, dropout=0.1, ffn_mult=4),
                       tokenizer_sha256=file_hash(token_root / "tokenizer.json")), checkpoint)
            cfg = configuration()
            cfg["train"].update(device="cpu", precision="fp32")
            cfg["progress"]["enabled"] = False
            align = cfg["alignment"]
            align.update(checkpoint=str(checkpoint), tokenizer_dir=str(token_root), source_dir=str(source),
                         output_dir=str(root / "alignment"), train_examples=4, dev_examples=2, test_examples=2,
                         max_new_tokens=80, max_prompt_tokens=80, min_words=6, keywords=vocabulary,
                         max_scan=100, batch_size=1, grad_accum=1, eval_every=1, save_every=1, stop_after=1)
            for method in ("sft", "dpo", "rlvr"):
                align[method].update(steps=2, warmup_steps=0)
            align["rlvr"]["group_size"] = 2
            prepared = run_alignment(cfg, "align_prepare")
            self.assertTrue(prepared["prepared"])
            data = root / "alignment" / "data"
            splits = {s:[json.loads(line) for line in (data / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "dev", "test")}
            pairs = {s:{tuple(sorted(row["words"])) for row in values} for s,values in splits.items()}
            self.assertFalse(pairs["train"] & pairs["test"])
            self.assertFalse(pairs["train"] & pairs["dev"])
            for values in splits.values():
                for row in values:
                    self.assertTrue(reward(row["chosen"], row["words"], 6)["success"])
                    self.assertLess(reward(row["rejected"], row["words"], 6)["keyword_coverage"], 1)
            self.assertFalse(run_alignment(cfg, "align")["completed"])
            align["stop_after"] = None
            result = run_alignment(cfg, "align")
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["comparison"]), 4)
            report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report["evaluation"][0]["samples"]), 2)
            original = torch.load(root / "alignment" / "sft" / "final.pt", weights_only=True)
            continuous_cfg = copy.deepcopy(cfg)
            continuous_cfg["alignment"]["output_dir"] = str(root / "continuous")
            run_alignment(continuous_cfg, "align")
            continuous = torch.load(root / "continuous" / "sft" / "final.pt", weights_only=True)
            for key in original["model"]:
                torch.testing.assert_close(original["model"][key], continuous["model"][key], rtol=0, atol=0)
            self.assertTrue(run_alignment(cfg, "align_eval")["completed"])
            from src.web_app import DemoService
            service = DemoService(cfg)
            response = service.generate(dict(model="base", mode="continuation", prompt="Once upon a time", max_new_tokens=3, seed=1))
            self.assertLessEqual(response["new_tokens"], 3)
            response = service.generate(dict(model="sft", mode="instruction", words=["cat", "tree"], max_new_tokens=3, seed=1))
            self.assertIn("verification", response)
            self.assertEqual(len(service.summary()["alignment"]["evaluation"]), 4)
            # RLVR 断点必须恢复独立的采样 generator，保持后续 on-policy 轨迹一致。
            stage_cfg = copy.deepcopy(align)
            stage_cfg.update(output_dir=str(root / "rl_resume"), stop_after=1)
            token_info = dict(instance=tokenizer, sha256=file_hash(token_root / "tokenizer.json"))
            manifest = prepared["data"]
            parent = root / "alignment" / "sft" / "final.pt"
            args = (cfg, stage_cfg, "rlvr", parent, manifest, token_info, read_examples(data / "train.jsonl"), read_examples(data / "dev.jsonl"))
            self.assertFalse(train_stage(*args)[1])
            stage_cfg["stop_after"] = None
            resumed_path, complete = train_stage(*args)
            self.assertTrue(complete)
            rl_resumed = torch.load(resumed_path, weights_only=True)
            rl_continuous = torch.load(root / "alignment" / "rlvr" / "final.pt", weights_only=True)
            for key in rl_resumed["model"]:
                torch.testing.assert_close(rl_resumed["model"][key], rl_continuous["model"][key], rtol=0, atol=0)

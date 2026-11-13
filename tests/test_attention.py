from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from src.attention_reference import error_metrics, naive_attention, reference_attention
from src.attention_tasks import make_cases, plot_results, run_attention, validate_config


def config():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path("configs").resolve())):
        return OmegaConf.to_container(compose(config_name="config"), resolve=True)["attention"]


class AttentionTests(unittest.TestCase):
    def test_reference_against_position_loop_and_causality(self):
        torch.manual_seed(7)
        q, k, v = [torch.randn(2, 3, 7, 8) for _ in range(3)]
        for causal in (True, False):
            expected = torch.empty_like(q)
            for b in range(2):
                for h in range(3):
                    for i in range(7):
                        end = i + 1 if causal else 7
                        weights = (k[b, h, :end] @ q[b, h, i] / 8 ** 0.5).softmax(0)
                        expected[b, h, i] = weights @ v[b, h, :end]
            torch.testing.assert_close(reference_attention(q, k, v, causal), expected)
            torch.testing.assert_close(naive_attention(q, k, v, causal), expected)
        changed = v.clone()
        changed[:, :, 4:] += 100
        torch.testing.assert_close(naive_attention(q, k, v)[:, :, :4],
                                   naive_attention(q, k, changed)[:, :, :4])

    def test_error_rejects_nonfinite_and_wrong_output(self):
        expected = torch.ones(3)
        self.assertTrue(error_metrics(expected, expected, 0.002, 0.01)["passed"])
        self.assertFalse(error_metrics(expected * 2, expected, 0.002, 0.01)["passed"])
        for value in (float("nan"), float("inf")):
            result = error_metrics(torch.full((3,), value), expected, 0.002, 0.01)
            self.assertFalse(result["passed"])
            self.assertIsNone(result["max_abs_error"])

    def test_hydra_plan_and_invalid_config(self):
        cfg = config()
        validate_config(cfg)
        self.assertEqual(len(make_cases(cfg)), 20)
        self.assertEqual({r["length"] for r in make_cases(cfg)}, {256, 512, 1024, 2048, 4096})
        for key, value in [("lengths", [0]), ("batch_sizes", [1, 1]),
                           ("head_dim", 96), ("repeat_ms", 0), ("dtypes", ["float32"])]:
            modified = dict(cfg, **{key: value})
            with self.assertRaises(ValueError):
                validate_config(modified)

    def test_plot_writes_image(self):
        rows = []
        for name, speed in (("pytorch_naive", 1), ("pytorch_flash_sdpa", 2), ("triton", 1.5)):
            for length in (256, 1024):
                rows.append(dict(batch=1, heads=12, head_dim=64, causal=True,
                                 dtype="float16", implementation=name, length=length,
                                 status="ok", latency_ms=length / speed / 1000,
                                 extra_peak_mib=length / speed, speedup_vs_naive=speed))
        with tempfile.TemporaryDirectory(dir="work") as directory:
            paths = plot_results(rows, Path(directory))
            self.assertEqual(len(paths), 1)
            self.assertGreater(Path(paths[0]).stat().st_size, 1000)

    @unittest.skipIf(torch.cuda.is_available(), "本检查用于无 GPU 的本地环境")
    def test_cpu_gives_actionable_message(self):
        with self.assertRaisesRegex(RuntimeError, "服务器"):
            run_attention(config(), {})


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernel 需要 CUDA 服务器")
class AttentionCudaTests(unittest.TestCase):
    def test_kernel_tails_causality_and_large_scores(self):
        from src.attention_kernel import triton_attention
        torch.manual_seed(123)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                for dtype in (torch.float16, torch.bfloat16):
                    for length, dim in ((1, 32), (63, 64), (65, 128), (129, 64), (256, 64)):
                        q, k, v = [torch.randn(2, 3, length, dim, device="cuda", dtype=dtype)
                                   for _ in range(3)]
                        for causal in (False, True):
                            expected = reference_attention(q, k, v, causal)
                            actual = triton_attention(q, k, v, causal=causal)
                            atol, rtol = (0.002, 0.01) if dtype == torch.float16 else (0.02, 0.05)
                            torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
                # 检查 online softmax 在较大但有限的分数下仍保持稳定。
                q, k, v = [torch.randn(1, 2, 129, 64, device="cuda", dtype=torch.float16)
                           for _ in range(3)]
                q, k = q * 5, k * 5
                torch.testing.assert_close(triton_attention(q, k, v).float(),
                                           reference_attention(q, k, v), atol=0.005, rtol=0.02)
            q.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "不能用于训练"):
                triton_attention(q, k, v)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    unittest.main()

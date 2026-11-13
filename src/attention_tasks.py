"""一条命令执行 Attention 前向/梯度检查、测速和结果导出。"""

import csv
from datetime import datetime, timezone
import itertools
import json
import logging
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.attention_reference import error_metrics, naive_attention, reference_attention, reference_gradients
from src.progress import progress_bar


def validate_config(cfg):
    if cfg["mode"] not in ("forward", "forward_backward"):
        raise ValueError("attention.mode 只支持 forward/forward_backward")
    for key in ("batch_sizes", "lengths", "edge_lengths", "edge_head_dims"):
        values = cfg[key]
        if not values or any(type(n) is not int or n < 1 for n in values):
            raise ValueError(f"attention.{key} 必须为非空正整数列表")
        if len(values) != len(set(values)):
            raise ValueError(f"attention.{key} 不应包含重复项")
    if not cfg["dtypes"] or any(d not in ("float16", "bfloat16") for d in cfg["dtypes"]):
        raise ValueError("attention.dtypes 只支持 float16/bfloat16")
    if len(cfg["dtypes"]) != len(set(cfg["dtypes"])):
        raise ValueError("attention.dtypes 不应包含重复项")
    for key in ("heads", "head_dim", "seed", "warmup_ms", "repeat_ms"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"attention.{key} 取值无效")
    if any(d not in (32, 64, 128) for d in [cfg["head_dim"], *cfg["edge_head_dims"]]):
        raise ValueError("head_dim 只支持 32、64、128")
    if type(cfg["causal"]) is not bool:
        raise ValueError("attention.causal 必须为布尔值")
    if cfg["block_m"] not in (32, 64, 128) or cfg["block_n"] not in (32, 64, 128):
        raise ValueError("block_m/block_n 只支持 32、64、128")
    if cfg["num_warps"] not in (4, 8):
        raise ValueError("num_warps 只支持 4、8")
    for field in ("tolerance", "gradient_tolerance"):
        for dtype in cfg["dtypes"]:
            tolerance = cfg[field][dtype]
            if any(not 0 < float(tolerance[key]) < 1 for key in ("atol", "rtol")):
                raise ValueError("误差容限必须在 (0, 1) 之间")


def make_cases(cfg):
    return [dict(batch=batch, heads=cfg["heads"], length=length,
                 head_dim=cfg["head_dim"], dtype=dtype, causal=cfg["causal"])
            for dtype, batch, length in itertools.product(
                cfg["dtypes"], cfg["batch_sizes"], cfg["lengths"])]


def make_inputs(case, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (case["batch"], case["heads"], case["length"], case["head_dim"])
    return tuple(torch.randn(shape, device=device, dtype=getattr(torch, case["dtype"]),
                             generator=generator) for _ in range(3))


def implementations(q, k, v, cfg, triton_attention):
    return {
        "pytorch_naive": lambda: naive_attention(q, k, v, cfg["causal"]),
        "pytorch_flash_sdpa": lambda: F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=cfg["causal"]),
        "triton": lambda: triton_attention(
            q, k, v, causal=cfg["causal"], block_m=cfg["block_m"],
            block_n=cfg["block_n"], num_warps=cfg["num_warps"]),
    }


def check_case(case, cfg, device, seed, triton_attention):
    q, k, v = make_inputs(case, device, seed)
    oracle = reference_attention(q, k, v, case["causal"])
    tolerance = cfg["tolerance"][case["dtype"]]
    checks = []
    with torch.no_grad():
        for name, function in implementations(q, k, v, cfg, triton_attention).items():
            output = function()
            checks.append(dict(case, implementation=name, component="output",
                               **error_metrics(output, oracle, **tolerance)))
            del output
    del oracle
    if cfg["mode"] == "forward_backward":
        generator = torch.Generator(device=device).manual_seed(seed + 100000)
        upstream = torch.randn(q.shape, dtype=q.dtype, device=device, generator=generator)
        expected = reference_gradients(q, k, v, upstream, case["causal"])
        tolerance = cfg["gradient_tolerance"][case["dtype"]]
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
            for name, function in implementations(*inputs, cfg, triton_attention).items():
                values = torch.autograd.grad(function(), inputs, upstream)
                for component, actual, reference in zip(("dQ", "dK", "dV"), values, expected):
                    checks.append(dict(case, implementation=name, component=component,
                                       **error_metrics(actual, reference, **tolerance)))
    return checks


def forward_backward(function, inputs, upstream):
    # autograd.grad 每次建立/释放新图，不累积 leaf.grad，不保留旧图。
    return torch.autograd.grad(function(), inputs, upstream)


def measure(function, device, cfg, do_bench):
    # 首次调用包含 JIT 编译，不纳入耗时或显存峰值。
    output = function()
    torch.cuda.synchronize(device)
    del output
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    output = function()
    torch.cuda.synchronize(device)
    extra_mib = (torch.cuda.max_memory_allocated(device) - baseline) / 2 ** 20
    del output
    # CUDA event 计时，返回中位数和 20/80 分位数，包含返回张量的分配。
    median, p20, p80 = do_bench(function, warmup=cfg["warmup_ms"],
                              rep=cfg["repeat_ms"], quantiles=[0.5, 0.2, 0.8])
    return dict(latency_ms=median, p20_ms=p20, p80_ms=p80, extra_peak_mib=extra_mib)


def save_report(root, report):
    # 最后一次成功写入始终完整；失败也保留已完成的检查/测速。
    temporary = root / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(root / "results.json")
    rows = report["benchmarks"]
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (root / "benchmarks.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def plot_results(rows, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(r["dtype"], r["batch"], r["heads"], r["head_dim"], r["causal"], r.get("phase", "forward"))
                     for r in rows})
    paths = []
    for dtype, batch, heads, dim, causal, phase in groups:
        selected = [r for r in rows if (r["dtype"], r["batch"], r["heads"],
                                       r["head_dim"], r["causal"], r.get("phase", "forward")) ==
                    (dtype, batch, heads, dim, causal, phase) and r["status"] == "ok"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for implementation in ("pytorch_naive", "pytorch_flash_sdpa", "triton"):
            points = sorted((r for r in selected if r["implementation"] == implementation),
                            key=lambda r: r["length"])
            if not points:
                continue
            lengths = [r["length"] for r in points]
            axes[0].plot(lengths, [r["latency_ms"] for r in points], "o-", label=implementation)
            axes[1].plot(lengths, [r["extra_peak_mib"] for r in points], "o-", label=implementation)
            axes[2].plot(lengths, [r["speedup_vs_naive"] for r in points], "o-", label=implementation)
        title = "Forward + backward latency" if phase == "forward_backward" else "Forward latency"
        for ax, title, ylabel in zip(axes, [title, "Extra peak allocated memory", "Speedup over naive"],
                                     ["Milliseconds", "MiB", "Times"]):
            ax.set(title=title, xlabel="Sequence length", ylabel=ylabel)
            ax.grid(alpha=0.25)
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{dtype} | B={batch}, H={heads}, D={dim} | causal={causal} | {phase}")
        fig.tight_layout()
        suffix = "_forward_backward" if phase == "forward_backward" else ""
        path = root / f"attention_{dtype}_b{batch}{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def run_attention(cfg, progress_cfg):
    validate_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("Attention 实验需要 Linux NVIDIA CUDA 服务器，请在服务器运行 stat=attention")
    device = torch.device(cfg["device"])
    if device.type != "cuda":
        raise ValueError("attention.device 必须指定 CUDA 设备")
    if "bfloat16" in cfg["dtypes"] and torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("此 GPU 不支持本实验需要的 BF16 Tensor Core，请用 FP16 配置")
    try:
        import triton
        from triton.testing import do_bench
        from src.attention_kernel import triton_attention
    except ImportError as exc:
        raise RuntimeError("缺少 Triton；在现有 Linux CUDA 环境安装 requirements-attention.txt，勿替换 PyTorch") from exc

    root = Path(cfg["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = make_cases(cfg)
    edge_cases = [dict(batch=1, heads=2, length=length, head_dim=dim,
                       dtype=dtype, causal=cfg["causal"])
                  for dtype, dim, length in itertools.product(
                      cfg["dtypes"], cfg["edge_head_dims"], cfg["edge_lengths"])]
    report = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg, "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device), "device": str(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }, "checks": [], "benchmarks": [], "plots": [], "issues": [],
        "notes": ["单卡 Attention 算子实验，无 dropout，支持一阶反向，未接入语言模型训练。",
                  "forward_backward 包含建图前向及 dQ/dK/dV，不含 loss、优化器或完整 Transformer。",
                  "反向重算分块概率，保存线性大小的 LSE 和输入输出，不保存 N×N 矩阵。",
                  "SDPA 强制 FLASH_ATTENTION 后端，不允许静默回退 math。",
                  "latency_ms 是预热后 CUDA event 计时中位数，不包含编译或正确性检查。",
                  "extra_peak_mib 是单次调用新增的 PyTorch allocated 峰值，包含输出，不包含输入、allocator reserved 或其他进程。",
                  "FP32 oracle 禁用 TF32；低精度按 atol + rtol * abs(reference) 判定。",
                  "分块避免保存 N×N 中间矩阵，但精确 Attention 的算术复杂度仍为 O(N²)。"],
    }
    started = time.perf_counter()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        save_report(root, report)
        # 强制 Flash backend：不把 math SDPA 的成绩误报成 FlashAttention。
        with torch.cuda.device(device), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            bar = progress_bar(edge_cases + cases, config=progress_cfg, desc="Attention check", unit="case")
            with bar:
                for index, case in enumerate(bar):
                    checks = check_case(case, cfg, device, cfg["seed"] + index, triton_attention)
                    report["checks"].extend(checks)
                    failures = [row for row in checks if not row["passed"]]
                    save_report(root, report)
                    if failures:
                        report["issues"].append(f"正确性检查失败：{failures}")
                        raise ValueError("Attention 误差超出容限，已停止测速；查看 results.json")
            logging.info("%s 正确性检查全部通过，开始 %d 个配置 × 3 种实现测速", cfg["mode"], len(cases))
            bar = progress_bar(cases, config=progress_cfg, desc="Attention benchmark", unit="case")
            with bar:
                for index, case in enumerate(bar):
                    q, k, v = make_inputs(case, device, cfg["seed"] + len(edge_cases) + index)
                    phases = ["forward"] if cfg["mode"] == "forward" else ["forward", "forward_backward"]
                    upstream = torch.ones_like(q)
                    for phase in phases:
                        training = phase == "forward_backward"
                        inputs = tuple(t.detach().requires_grad_(training) for t in (q, k, v))
                        baseline_ms = None
                        with torch.set_grad_enabled(training):
                            for name, function in implementations(*inputs, cfg, triton_attention).items():
                                call = (lambda: forward_backward(function, inputs, upstream)) if training else function
                                row = dict(case, implementation=name, phase=phase, status="ok")
                                row.update(measure(call, device, cfg, do_bench))
                                if name == "pytorch_naive":
                                    baseline_ms = row["latency_ms"]
                                row["speedup_vs_naive"] = baseline_ms / row["latency_ms"]
                                report["benchmarks"].append(row)
                                del call
                    # function 的闭包也持有 Q/K/V，离开本配置后释放。
                    del function, inputs, upstream, q, k, v
                    save_report(root, report)
                    bar.set_postfix(dtype=case["dtype"], B=case["batch"], N=case["length"])
            report["plots"] = plot_results(report["benchmarks"], root)
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["issues"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        report["elapsed_seconds"] = time.perf_counter() - started
        save_report(root, report)
    return dict(output_dir=str(root), checks=len(report["checks"]),
                benchmark_rows=len(report["benchmarks"]), passed=True,
                results=str(root / "results.json"), csv=str(root / "benchmarks.csv"),
                plots=report["plots"])

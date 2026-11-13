"""数据质量对照 + 四档模型 scaling + 独立大模型外推验证。"""

import copy
import csv
import gc
import json
import logging
import math
from pathlib import Path
import re
import time

import torch

from src.data import write_json
from src.model import GPT
from src.progress import progress_bar
from src.science_data import encode_corpora, fingerprint, prepare_corpora
from src.science_fit import predict, summarize_scaling
from src.token_data import TokenStream
from src.training import append_json, evaluate, learning_rate, setup, synchronize, update


def validate_science(cfg):
    for key in ("train_documents", "val_documents", "token_budget", "batch_size", "grad_accum", "context",
                "eval_every", "save_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"science.{key} 必须为正整数")
    if not cfg["seeds"] or any(type(seed) is not int or seed < 0 for seed in cfg["seeds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("science.seeds 必须为不重复的非负整数列表")
    per_step = cfg["batch_size"] * cfg["grad_accum"] * cfg["context"]
    if cfg["token_budget"] % per_step:
        raise ValueError("science.token_budget 必须为 batch_size*grad_accum*context 的整数倍")
    if not 0 <= cfg["warmup_steps"] < cfg["token_budget"] // per_step:
        raise ValueError("science.warmup_steps 必须小于总步数")
    if cfg["stop_after"] is not None and (type(cfg["stop_after"]) is not int or cfg["stop_after"] < 1):
        raise ValueError("science.stop_after 必须为正整数或 null")
    if not 0 < cfg["min_lr"] <= cfg["lr"] or not 0 <= cfg["dropout"] < 1:
        raise ValueError("science 学习率或 dropout 无效")
    if cfg["source_format"] not in ("jsonl", "wet"):
        raise ValueError("science.source_format 只支持 jsonl/wet")
    if len(cfg["models"]) != 4:
        raise ValueError("本实验要求四个不同规模用于拟合")
    specs = [*cfg["models"], cfg["holdout_model"]]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("模型名必须唯一且只包含字母、数字、下划线或连字符")
    if cfg["comparison_model"] not in names[:-1]:
        raise ValueError("comparison_model 必须为四个拟合模型之一")
    for spec in specs:
        if any(type(spec[k]) is not int or spec[k] < 1 for k in ("layers", "width", "heads")) or spec["width"] % spec["heads"]:
            raise ValueError("模型维度无效或 width 不能被 heads 整除")
    clean = cfg["cleaning"]
    for key in ("strip_html", "language_filter", "quality_filter", "redact_pii", "exact_dedup", "near_dedup"):
        if type(clean[key]) is not bool:
            raise ValueError(f"cleaning.{key} 必须为布尔值")
    for key in ("min_chars", "max_chars", "shingle_words", "permutations", "bands"):
        if type(clean[key]) is not int or clean[key] < 1:
            raise ValueError(f"cleaning.{key} 必须为正整数")
    if clean["max_chars"] < clean["min_chars"] or clean["permutations"] % clean["bands"]:
        raise ValueError("清洗长度范围或 MinHash 分块配置无效")
    for key in ("min_latin_ratio", "max_symbol_ratio", "min_unique_word_ratio", "jaccard_threshold"):
        if not 0 < clean[key] <= 1:
            raise ValueError(f"cleaning.{key} 必须在 (0,1] 内")
    fit = cfg["fit"]
    if not 0 < fit["alpha_min"] < fit["alpha_max"] or fit["grid_points"] < 10 or fit["bootstrap_samples"] < 1:
        raise ValueError("幂律拟合配置无效")


def experiment_runs(cfg):
    models = {m["name"]: m for m in cfg["models"]}
    comparison = models[cfg["comparison_model"]]
    order = [comparison, *[m for m in cfg["models"] if m != comparison], cfg["holdout_model"]]
    return [dict(id=f"{model['name']}_{corpus}_s{seed}", model=model, corpus=corpus, seed=seed)
            for seed in cfg["seeds"]
            for model, corpus in [(comparison, "raw"), *[(m, "clean") for m in order]]]


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run_model(base_cfg, cfg, spec, token_root, token_meta, vocab_size):
    options = copy.deepcopy(base_cfg)
    train = options["train"]
    steps = cfg["token_budget"] // (cfg["batch_size"] * cfg["grad_accum"] * cfg["context"])
    train.update(seed=spec["seed"], batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
                 max_steps=steps, warmup_steps=cfg["warmup_steps"], lr=cfg["lr"], min_lr=cfg["min_lr"])
    device, precision = setup(options)
    model_cfg = {key: value for key, value in spec["model"].items() if key != "name"}
    model_cfg.update(context=cfg["context"], dropout=cfg["dropout"], ffn_mult=4, vocab_size=vocab_size)
    signature = fingerprint(dict(version=1, tokens=token_meta["key"], model=model_cfg,
                                 corpus=spec["corpus"], seed=spec["seed"], precision=precision,
                                 device_type=device.type, torch=str(torch.__version__),
                                 train={key: train[key] for key in ("max_steps", "warmup_steps", "lr", "min_lr", "batch_size", "grad_accum",
                                                                    "weight_decay", "beta1", "beta2", "grad_clip")},
                                 eval_every=cfg["eval_every"]))
    output = Path(cfg["output_dir"]).resolve() / "models" / spec["id"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature:
            raise ValueError(f"{spec['id']} 的结果配置已改变，请换 science.output_dir")
        logging.info("跳过已完成实验 %s", spec["id"])
        return result
    last = output / "last.pt"
    if any(output.iterdir()) and not last.exists():
        raise ValueError(f"{output} 不完整且没有 last.pt，请检查或使用新输出目录")
    model = GPT(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"],
                                 betas=(train["beta1"], train["beta2"]), weight_decay=train["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    rng = torch.Generator().manual_seed(spec["seed"])
    streams = {name: TokenStream(token_root / f"{name}.bin", token_meta["dtype"], cfg["context"])
               for name in (spec["corpus"], "val")}
    start, best, train_seconds = 0, float("inf"), 0.0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError(f"{spec['id']} 断点配置不匹配")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.set_state(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start, best, train_seconds = state["step"], state["best_val_loss"], state["train_seconds"]
        del state
    write_json(output / "config.json", dict(signature=signature, model=model_cfg, train=train, spec=spec))
    end = min(steps, cfg["stop_after"] or steps)
    if end <= start and end < steps:
        raise ValueError("stop_after 已达到；正式续训请设 science.stop_after=null")
    if not last.exists():
        atomic_checkpoint(last, dict(version=1, signature=signature, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=0,
                          best_val_loss=best, rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                          train_seconds=0.0))
    logging.info("实验 %s：%s 参数，%s/%s 步，%s token 预算", spec["id"], model.parameter_count(), start, steps, cfg["token_budget"])
    final_evaluation = None
    with progress_bar(total=steps, initial=start, config=base_cfg["progress"], desc=spec["id"], unit="step") as bar:
        for index in range(start, end):
            lr = learning_rate(index, train)
            for group in optimizer.param_groups:
                group["lr"] = lr
            synchronize(device)
            started = time.perf_counter()
            loss = update(model, streams[spec["corpus"]], optimizer, scaler, rng, train, device, precision)
            synchronize(device)
            train_seconds += time.perf_counter() - started
            step = index + 1
            record = dict(step=step, train_loss=loss, lr=lr)
            should_eval = step % cfg["eval_every"] == 0 or step == end
            if should_eval:
                final_evaluation = evaluate(model, streams["val"], cfg["batch_size"], device, precision, None)
                record["val_loss"] = final_evaluation["loss"]
                if final_evaluation["loss"] < best:
                    best = final_evaluation["loss"]
                    atomic_checkpoint(output / "best.pt", dict(version=1, signature=signature,
                                      model=model.state_dict(), model_config=model_cfg, step=step,
                                      val_loss=best, tokenizer_sha256=token_meta["tokenizer_sha256"]))
            append_json(output / "metrics.jsonl", record)
            if should_eval or step % cfg["save_every"] == 0:
                atomic_checkpoint(last, dict(version=1, signature=signature, model=model.state_dict(),
                                  optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
                                  best_val_loss=best, rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                                  cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                                  train_seconds=train_seconds))
            bar.set_postfix(loss=f"{loss:.4f}", val=f"{best:.4f}" if math.isfinite(best) else "--", refresh=False)
            bar.update(1)
    if end < steps:
        return dict(completed=False, run=spec["id"], step=end, total_steps=steps)
    # 断点若已经完成但尚未写 result，重新评估末尾模型即可。
    if final_evaluation is None:
        final_evaluation = evaluate(model, streams["val"], cfg["batch_size"], device, precision, None)
    result = dict(completed=True, signature=signature, run=spec["id"], model=spec["model"]["name"],
                  corpus=spec["corpus"], seed=spec["seed"], parameters=model.parameter_count(),
                  steps=steps, seen_tokens=cfg["token_budget"], final_val_loss=final_evaluation["loss"],
                  best_val_loss=best, val_tokens=final_evaluation["tokens"], train_seconds=train_seconds,
                  sampled_epoch_equivalent=cfg["token_budget"] / (len(streams[spec["corpus"]].tokens) - 1),
                  token_key=token_meta["key"], precision=precision, torch=str(torch.__version__),
                  cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    write_json(result_path, result)
    del model, optimizer, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def analyze(cfg, manifest, rows):
    if any(r["seen_tokens"] != cfg["token_budget"] or not r["completed"] for r in rows):
        raise ValueError("实验未完成指定 token 预算")
    if len({r["seen_tokens"] for r in rows}) != 1 or len({r["val_tokens"] for r in rows}) != 1 or len({r["token_key"] for r in rows}) != 1:
        raise ValueError("实验结果没有共享训练预算、验证集或分词器")
    comparisons = []
    for seed in cfg["seeds"]:
        pair = {r["corpus"]: r for r in rows if r["model"] == cfg["comparison_model"] and r["seed"] == seed}
        if set(pair) != {"raw", "clean"}:
            raise ValueError("清洗对照缺少相同种子的配对结果")
        comparisons.append(dict(seed=seed, raw_loss=pair["raw"]["final_val_loss"], clean_loss=pair["clean"]["final_val_loss"],
                                clean_minus_raw=pair["clean"]["final_val_loss"] - pair["raw"]["final_val_loss"]))
    scaling = summarize_scaling(rows, [m["name"] for m in cfg["models"]], cfg["holdout_model"]["name"], cfg["fit"], cfg["seeds"][0])
    root = Path(cfg["output_dir"]).resolve()
    report = dict(completed=True, data=manifest, data_quality=comparisons, scaling=scaling,
                  metric="final validation loss after identical processed-token budget; best loss is secondary",
                  warnings=["清洗不保证降低 loss；默认 TinyStories 输入已做过基础过滤，不能称为原始网页清洗实验。",
                            "清洗同时改变语料组成和唯一文本数量；等 token 预算不隔离所有影响因素。",
                            "LSH 近重复检索有漏检概率；语言/PII 是启发式，不替代 fastText 或完整个人信息检测。"])
    write_json(root / "report.json", report)
    with (root / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["plots"] = plot_science(root, manifest, comparisons, scaling, rows)
    write_json(root / "report.json", report)
    return dict(output_dir=str(root), completed=True, experiments=len(rows),
                mean_clean_minus_raw=sum(r["clean_minus_raw"] for r in comparisons) / len(comparisons),
                holdout=scaling["holdout"], report=str(root / "report.json"),
                csv=str(root / "experiments.csv"), plots=report["plots"], fit_warnings=scaling["fit"]["warnings"])


def plot_science(root, manifest, comparisons, scaling, rows):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Before", "After"], [manifest["statistics"]["raw"], manifest["statistics"]["clean"]])
    axes[0].set(title="Additional cleaning", ylabel="Training documents")
    for pair in comparisons:
        axes[1].plot(["Before", "After"], [pair["raw_loss"], pair["clean_loss"]], "o-", label=f"seed {pair['seed']}")
    axes[1].set(title="Equal-token paired comparison", ylabel="Final validation loss")
    axes[1].legend()
    fig.tight_layout()
    path = root / "data_quality.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    fig, ax = plt.subplots(figsize=(8, 4.7))
    sizes = np.array(scaling["parameters"])
    ax.errorbar(sizes, scaling["mean_loss"], yerr=scaling["std_loss"], fmt="o", capsize=3, label="Small models: mean +/- seed std")
    target = scaling["holdout"]
    grid = np.geomspace(sizes.min(), target["parameters"], 200)
    ax.plot(grid, predict(scaling["fit"], grid), label="Power-law candidate (fixed token budget)")
    ax.scatter([target["parameters"]], [target["predicted_loss"]], marker="x", s=65, label="Held-out prediction")
    ax.scatter([target["parameters"]], [target["actual_loss"]], marker="*", s=100, label="Held-out measured")
    ax.set(xscale="log", xlabel="Actual parameter count", ylabel="Final validation loss", title="Fixed-token parameter scaling; held-out large model")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = root / "scaling_law.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        metrics_path = root / "models" / row["run"] / "metrics.jsonl"
        metrics = {r["step"]: r for r in map(json.loads, metrics_path.read_text(encoding="utf-8").splitlines())}
        points = sorted((step, r["val_loss"]) for step, r in metrics.items() if "val_loss" in r)
        if points:
            ax.plot(*zip(*points), label=row["run"], alpha=0.75)
    ax.set(xlabel="Optimizer step", ylabel="Validation loss", title="Experiment validation curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = root / "validation_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run_science(base_cfg, task):
    cfg = base_cfg["science"]
    validate_science(cfg)
    root, manifest, tokenizer, tokenizer_sha = prepare_corpora(cfg, base_cfg["progress"])
    token_root, token_meta = encode_corpora(root, manifest, tokenizer, tokenizer_sha, Path(cfg["output_dir"]).resolve(), cfg["context"], base_cfg["progress"])
    runs = experiment_runs(cfg)
    parameter_counts = {}
    for spec in [*cfg["models"], cfg["holdout_model"]]:
        with torch.device("meta"):
            model = GPT(tokenizer.vocab_size, spec["layers"], spec["width"], spec["heads"], cfg["context"], cfg["dropout"])
        parameter_counts[spec["name"]] = model.parameter_count()
    sizes = [parameter_counts[m["name"]] for m in cfg["models"]]
    if len(set(sizes)) != 4 or parameter_counts[cfg["holdout_model"]["name"]] <= max(sizes):
        raise ValueError("四档模型参数量必须不同，且外推模型必须更大")
    output = Path(cfg["output_dir"]).resolve()
    plan = dict(experiments=len(runs), steps_per_run=cfg["token_budget"] // (cfg["batch_size"] * cfg["grad_accum"] * cfg["context"]),
                processed_tokens_per_run=cfg["token_budget"], seeds=cfg["seeds"], runs=runs,
                data=manifest, tokens=token_meta, tokenizer_sha256=tokenizer_sha, parameters=parameter_counts)
    write_json(output / "plan.json", plan)
    if task == "science_prepare":
        return dict(output_dir=str(output), prepared=True, experiments=len(runs), steps_per_run=plan["steps_per_run"],
                    statistics=manifest["statistics"], tokens=token_meta["files"])
    rows = []
    for spec in runs:
        if task == "science_analyze":
            path = output / "models" / spec["id"] / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            saved = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
            if (saved["spec"] != spec or result["token_key"] != token_meta["key"] or
                    result["parameters"] != parameter_counts[spec["model"]["name"]] or
                    saved["model"]["dropout"] != cfg["dropout"] or saved["model"]["context"] != cfg["context"]):
                raise ValueError("分析配置与已完成实验不一致")
            controlled = dict(batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"], lr=cfg["lr"],
                              min_lr=cfg["min_lr"], warmup_steps=cfg["warmup_steps"])
            controlled.update({key: base_cfg["train"][key] for key in ("weight_decay", "beta1", "beta2", "grad_clip")})
            if any(saved["train"][key] != value for key, value in controlled.items()):
                raise ValueError("分析时训练超参数与已有结果不一致")
            rows.append(result)
        else:
            result = run_model(base_cfg, cfg, spec, token_root, token_meta, tokenizer.vocab_size)
            if not result["completed"]:
                return result
            rows.append(result)
    return analyze(cfg, manifest, rows)

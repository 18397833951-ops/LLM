"""单卡训练闭环；默认模型约 0.1B，benchmark 也使用相同配置。"""

from contextlib import nullcontext
import csv
import json
import logging
import math
from pathlib import Path
import time

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data import file_hash, write_json
from src.model import GPT
from src.progress import progress_bar
from src.token_data import TokenStream, prepare_tokens
from src.tokenizer import BPETokenizer


def setup(cfg):
    train = cfg["train"]
    if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 4):
        raise RuntimeError("模型任务要求 PyTorch >= 2.4；数据与分词器任务不受影响")
    for name in ("batch_size", "grad_accum", "max_steps", "log_every", "eval_every", "save_every", "sample_every"):
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError(f"train.{name} 必须为正整数")
    if not 0 < train["min_lr"] <= train["lr"] or train["grad_clip"] <= 0:
        raise ValueError("学习率或 grad_clip 无效")
    if type(train["warmup_steps"]) is not int or not 0 <= train["warmup_steps"] < train["max_steps"]:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < max_steps")
    if train["stop_after"] is not None and (type(train["stop_after"]) is not int or train["stop_after"] < 1):
        raise ValueError("stop_after 必须为正整数或 null")
    device = torch.device(train["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("目前只支持 cpu 或 cuda 设备")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查服务器 PyTorch 环境")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    torch.manual_seed(train["seed"])
    precision = train["precision"]
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "fp32")
    if precision not in ("fp32", "fp16", "bf16") or (device.type == "cpu" and precision != "fp32"):
        raise ValueError("precision 支持 auto/fp32/fp16/bf16；CPU 使用 fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU 不支持 BF16")
    return device, precision


def amp(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def learning_rate(step, cfg):
    if cfg["warmup_steps"] and step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"] - 1)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(1, progress)))


def update(model, stream, optimizer, scaler, rng, cfg, device, precision):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    for _ in range(cfg["grad_accum"]):
        x, y = stream.sample(cfg["batch_size"], rng, device)
        with amp(device, precision):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练 loss 非有限值，已停止更新")
        total_loss += float(loss.detach())
        scaler.scale(loss / cfg["grad_accum"]).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("梯度非有限值，已停止更新；FP16 可尝试 BF16 或 FP32")
    scaler.step(optimizer)
    scaler.update()
    return total_loss / cfg["grad_accum"]


@torch.no_grad()
def evaluate(model, stream, batch_size, device, precision, max_tokens):
    was_training = model.training
    model.eval()
    total_loss = total_tokens = 0
    try:
        for x, y in stream.batches(batch_size, device, max_tokens):
            with amp(device, precision):
                _, loss = model(x, y)
            total_loss += float(loss) * y.numel()
            total_tokens += y.numel()
    finally:
        model.train(was_training)
    loss = total_loss / total_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("验证 loss 非有限值")
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 80 else None, "tokens": total_tokens}


@torch.no_grad()
def generate(model, tokenizer, cfg, device, precision):
    if type(cfg["max_new_tokens"]) is not int or cfg["max_new_tokens"] < 1:
        raise ValueError("generate.max_new_tokens 必须是正整数")
    if cfg["temperature"] < 0 or type(cfg["top_k"]) is not int or cfg["top_k"] < 0:
        raise ValueError("temperature 和 top_k 必须非负，top_k 必须为整数")
    ids = tokenizer.encode(cfg["prompt"], add_bos=False)
    if not ids:
        ids = [tokenizer.bos_id]
    input_length = len(ids)
    was_training = model.training
    model.eval()
    # 采样不会消耗训练用的全局 RNG，保存前后不改变后续训练轨迹。
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["seed"])
            for _ in range(cfg["max_new_tokens"]):
                x = torch.tensor([ids[-model.context:]], dtype=torch.long, device=device)
                with amp(device, precision):
                    logits, _ = model(x)
                scores = logits[0, -1].float()
                scores[[tokenizer.pad_id, tokenizer.bos_id]] = float("-inf")
                if cfg["temperature"] == 0:
                    token = int(scores.argmax())
                else:
                    scores = scores / cfg["temperature"]
                    if cfg["top_k"]:
                        threshold = torch.topk(scores, min(cfg["top_k"], scores.numel())).values[-1]
                        scores = scores.masked_fill(scores < threshold, float("-inf"))
                    probabilities = scores.softmax(dim=-1)
                    token = int(torch.multinomial(probabilities, 1))
                ids.append(token)
                if token == tokenizer.eos_id:
                    break
    finally:
        model.train(was_training)
    return {"prompt": cfg["prompt"], "text": tokenizer.decode(ids, errors="replace"),
            "new_tokens": len(ids) - input_length}


def save_checkpoint(path, model, optimizer, scaler, rng, step, best, model_cfg, cfg, meta, precision):
    value = {"version": 1, "model": model.state_dict(), "model_config": model_cfg,
             "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
             "step": step, "seen_tokens": step * cfg["train"]["batch_size"] * cfg["train"]["grad_accum"] * model.context,
             "best_val_loss": best, "config": cfg, "data_key": meta["key"], "precision": precision,
             "tokenizer_sha256": meta["signature"]["tokenizer_sha256"]}
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    return checkpoint


def restore(checkpoint, model, optimizer, scaler, rng, model_cfg, cfg, meta, precision):
    if checkpoint["model_config"] != model_cfg or checkpoint["data_key"] != meta["key"]:
        raise ValueError("续训模型或数据与 checkpoint 不一致")
    keys = ("seed", "batch_size", "grad_accum", "max_steps", "lr", "min_lr", "warmup_steps",
            "weight_decay", "beta1", "beta2", "grad_clip")
    if any(checkpoint["config"]["train"][key] != cfg["train"][key] for key in keys) or checkpoint["precision"] != precision:
        raise ValueError("续训须保持原训练预算、优化参数及精度；stop_after 和日志间隔可调整")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng.set_state(checkpoint["rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint["step"], checkpoint["best_val_loss"]


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def benchmark_or_train(cfg, task, run_dir):
    device, precision = setup(cfg)
    train = cfg["train"]
    output = Path(train["output_dir"]).resolve()
    if task == "train" and output.exists() and not train["resume"]:
        raise FileExistsError(f"训练目录已存在：{output}；续训请指定 train.resume，或使用新 output_dir")
    tokenizer, token_dir, meta = prepare_tokens(cfg)
    model_cfg = {"vocab_size": tokenizer.vocab_size, **cfg["model"]}
    model = GPT(**model_cfg).to(device)
    logging.info("模型参数 %s (%.4fB)，device=%s，precision=%s", model.parameter_count(), model.parameter_count()/1e9, device, precision)
    streams = {split: TokenStream(token_dir / f"{split}.bin", meta["dtype"], model.context) for split in ("train", "val")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"],
                                 betas=(train["beta1"], train["beta2"]), weight_decay=train["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    rng = torch.Generator().manual_seed(train["seed"])
    tokens_per_step = train["batch_size"] * train["grad_accum"] * model.context

    if task == "benchmark":
        bench = cfg["benchmark"]
        if any(type(bench[key]) is not int or bench[key] < minimum for key, minimum in (("warmup_steps", 0), ("steps", 1), ("estimate_tokens", 1))):
            raise ValueError("benchmark 步数与 token 预算无效")
        for _ in progress_bar(range(bench["warmup_steps"]), config=cfg.get("progress"), desc="Warmup", unit="step"):
            update(model, streams["train"], optimizer, scaler, rng, train, device, precision)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with progress_bar(total=bench["steps"], config=cfg.get("progress"), desc="Benchmark", unit="step") as progress:
            started = time.perf_counter()
            for _ in range(bench["steps"]):
                loss = update(model, streams["train"], optimizer, scaler, rng, train, device, precision)
                progress.set_postfix(loss=f"{loss:.4f}", refresh=False)
                progress.update(1)
            synchronize(device)
            elapsed = time.perf_counter() - started
        speed = bench["steps"] * tokens_per_step / elapsed
        result = {"parameters": model.parameter_count(), "model_config": model_cfg,
                  "device": str(device), "precision": precision, "torch_version": str(torch.__version__),
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "seconds": elapsed, "tokens_per_second": speed, "last_loss": loss,
                  "seconds_per_step": elapsed / bench["steps"], "tokens_per_step": tokens_per_step,
                  "estimated_training_hours": train["max_steps"] * tokens_per_step / speed / 3600,
                  "estimate_budget_tokens": bench["estimate_tokens"],
                  "estimate_budget_hours": bench["estimate_tokens"] / speed / 3600,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                  "note": "仅训练步骤测速，包含取 batch 和优化更新；不含首次编码、验证、生成和保存，未保存模型权重"}
        write_json(Path(run_dir) / "benchmark.json", {**result, "config": cfg})
        return result

    start_step, best = 0, float("inf")
    if train["resume"]:
        checkpoint = load_checkpoint(train["resume"])
        start_step, best = restore(checkpoint, model, optimizer, scaler, rng, model_cfg, cfg, meta, precision)
        if Path(train["resume"]).resolve().parent != output:
            raise ValueError("续训 output_dir 必须与 checkpoint 所在目录一致")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg)
    write_json(output / "model_info.json", {"parameters": model.parameter_count(), "model_config": model_cfg, "data": meta})
    end_step = min(train["max_steps"], train["stop_after"] or train["max_steps"])
    if start_step >= end_step:
        raise ValueError("checkpoint 已达到本次目标步数，请检查 stop_after/max_steps")
    if not start_step:
        logging.info("正在进行初始验证和生成样例")
        initial = evaluate(model, streams["val"], train["batch_size"], device, precision, cfg["eval"]["max_tokens"])
        append_json(output / "metrics.jsonl", {"step": 0, "val_loss": initial["loss"], "val_tokens": initial["tokens"]})
        append_json(output / "samples.jsonl", {"step": 0, **generate(model, tokenizer, cfg["generate"], device, precision)})
    measured_tokens = measured_seconds = 0
    with logging_redirect_tqdm(), progress_bar(total=train["max_steps"], initial=start_step,
                                               config=cfg.get("progress"), desc="Train", unit="step") as progress:
        for step in range(start_step, end_step):
            lr = learning_rate(step, train)
            for group in optimizer.param_groups:
                group["lr"] = lr
            synchronize(device)
            started = time.perf_counter()
            loss = update(model, streams["train"], optimizer, scaler, rng, train, device, precision)
            synchronize(device)
            elapsed = time.perf_counter() - started
            measured_tokens += tokens_per_step
            measured_seconds += elapsed
            completed = step + 1
            record = {"step": completed, "seen_tokens": completed * tokens_per_step,
                      "train_loss": loss, "lr": lr, "tokens_per_second": tokens_per_step / elapsed}
            is_best = False
            if completed % train["eval_every"] == 0 or completed == end_step:
                progress.set_description("Train / validation")
                evaluation = evaluate(model, streams["val"], train["batch_size"], device, precision, cfg["eval"]["max_tokens"])
                record.update(val_loss=evaluation["loss"], val_tokens=evaluation["tokens"])
                is_best = evaluation["loss"] < best
                best = min(best, evaluation["loss"])
            append_json(output / "metrics.jsonl", record)
            if completed % train["log_every"] == 0 or completed == end_step:
                logging.info("step %s/%s loss %.4f val %s tokens/s %.0f", completed, train["max_steps"], loss, record.get("val_loss"), measured_tokens/measured_seconds)
            if completed % train["sample_every"] == 0 or completed == end_step:
                progress.set_description("Train / sampling")
                append_json(output / "samples.jsonl", {"step": completed, **generate(model, tokenizer, cfg["generate"], device, precision)})
            if completed % train["save_every"] == 0 or completed == end_step or is_best:
                progress.set_description("Train / saving")
                save_checkpoint(output / "last.pt", model, optimizer, scaler, rng, completed, best, model_cfg, cfg, meta, precision)
            if is_best:
                save_checkpoint(output / "best.pt", model, optimizer, scaler, rng, completed, best, model_cfg, cfg, meta, precision)
            progress.set_description("Train", refresh=False)
            progress.set_postfix(loss=f"{loss:.4f}", best_val=f"{best:.4f}" if math.isfinite(best) else "--",
                                 tok_s=f"{measured_tokens/measured_seconds:.0f}", refresh=False)
            progress.update(1)
    return {"output_dir": str(output), "step": end_step, "seen_tokens": end_step*tokens_per_step,
            "best_val_loss": best, "parameters": model.parameter_count(), "completed_plan": end_step == train["max_steps"]}


def plot_training(cfg):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    output = Path(cfg["train"]["output_dir"])
    # 恢复后可能重放部分步骤，同一步保留日志中最后一次记录。
    rows = {row["step"]: row for row in (json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = [rows[key] for key in sorted(rows)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in (("train_loss", "Train"), ("val_loss", "Validation")):
        points = [(row["step"], row[key]) for row in rows if key in row]
        if points:
            ax.plot(*zip(*points), label=label)
    ax.set(xlabel="Optimizer step", ylabel="Cross-entropy (nats)", title="Language model training")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)
    return {"curve": str(output / "loss_curve.png"), "csv": str(output / "metrics.csv")}


def run_task(cfg, task, run_dir):
    if task == "plot":
        return plot_training(cfg)
    if task in ("benchmark", "train"):
        return benchmark_or_train(cfg, task, run_dir)
    device, precision = setup(cfg)
    options = cfg["eval"] if task == "eval" else cfg["generate"]
    checkpoint = load_checkpoint(options["checkpoint"])
    tokenizer_path = Path(cfg["tokenizer"]["output_dir"]) / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    # 路径可变化，但词表文件内容必须与训练时一致。
    tokenizer_sha = file_hash(tokenizer_path)
    if checkpoint["tokenizer_sha256"] != tokenizer_sha:
        raise ValueError("词表与 checkpoint 不一致")
    model = GPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    if task == "generate":
        result = generate(model, tokenizer, cfg["generate"], device, precision)
    else:
        _, token_dir, meta = prepare_tokens(cfg)
        if meta["key"] != checkpoint["data_key"]:
            raise ValueError("评估数据与 checkpoint 来源不同")
        stream = TokenStream(token_dir / "val.bin", meta["dtype"], model.context)
        result = evaluate(model, stream, cfg["train"]["batch_size"], device, precision, cfg["eval"]["max_tokens"])
    write_json(Path(run_dir) / f"{task}.json", result)
    return result

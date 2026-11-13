"""从服务器真实结果导出英文报告草稿；缺失结果明确留空。"""

from pathlib import Path
import shutil

from src.web_app import DemoService


def metric(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (float, int)) else "not available"


def export_report(cfg):
    service = DemoService(cfg)
    state = service.summary()
    root = Path(cfg["alignment"]["output_dir"]).resolve().parent / "submission"
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    train, attention, science, alignment = (state[key] for key in ("training", "attention", "science", "alignment"))
    lines = ["# A Small Story Language Model: Pretraining, Attention Kernels, Scaling, and Post-Training", "",
             "> Automatically assembled technical report draft from local experimental artifacts. Replace author details and apply the required NeurIPS template before submission. Missing data are not fabricated.", "",
             "## Abstract", "",
             "We study an English story language model implemented from scratch, including a byte-level BPE tokenizer and decoder-only Transformer. We evaluate a tiled Triton attention implementation, fixed-token parameter scaling, and keyword-conditioned post-training. The scope is a single-GPU educational study, not a general chat assistant or a reproduction of every original assignment component.", "",
             "## 1. Problem and motivation", "",
             "The goal is to connect implementation choices to measurable language-model behavior: how a model learns to continue stories, how attention kernels affect isolated operator cost, how parameter scale predicts loss under a controlled budget, and whether post-training improves explicit word constraints.", "",
             "## 2. From-scratch pretraining", "",
             "The implementation uses handwritten BPE, learned positional embeddings, pre-normalization, residual connections, a GELU feed-forward network, causal self-attention, and tied input/output embeddings. Training data are TinyStories. The preprocessing keeps official train/validation provenance and file checksums.", "",
             f"Recorded parameter count: {train.get('parameters', 'not available')}. Recorded optimizer steps: {train.get('steps', 'not available')}. Best validation loss: {metric(train.get('best_val_loss'))}; corresponding perplexity: {metric(train.get('perplexity'))}.", "",
             "## 3. Attention implementation", "",
             "The Triton implementation uses tiled computation and online softmax with FP32 accumulation. Its first-order backward recomputes block probabilities using saved row log-sum-exp values, without storing the full quadratic score matrix. We compare against a materialized PyTorch reference and an explicitly selected Flash SDPA backend.", ""]
    if attention:
        lines.append(f"Status: {attention['status']}; comparisons: {attention['checks']}; failed comparisons: {attention['failed_checks']}. Timings below are isolated attention calls, not end-to-end model training speedups.")
        lines += ["", "| Precision | Batch | Length | Phase | Triton / naive speedup | Triton / SDPA latency ratio |", "|---|---:|---:|---|---:|---:|"]
        rows = attention["benchmarks"]
        if rows:
            longest = max(r["length"] for r in rows)
            for row in rows:
                if row["implementation"] == "triton" and row["length"] == longest:
                    sdpa = next((r for r in rows if r["implementation"] == "pytorch_flash_sdpa" and
                                 all(r.get(k, "forward") == row.get(k, "forward") for k in ("phase", "dtype", "batch", "length"))), None)
                    ratio = row["latency_ms"] / sdpa["latency_ms"] if sdpa else None
                    lines.append(f"| {row['dtype']} | {row['batch']} | {row['length']} | {row.get('phase','forward')} | {metric(row['speedup_vs_naive'],2)} | {metric(ratio,2)} |")
    else:
        lines.append("Attention results are not available in the configured directory.")
    lines += ["", "## 4. Data quality and parameter scaling", "",
              "We compare models at an identical processed-token budget with a frozen shared tokenizer and common validation stream. Four smaller scales are fitted using L(N) = c + a (N/1e6)^(-alpha); the larger model is held out from the fit. Reported comparison losses are final-budget losses, not independently selected best steps.", ""]
    if science:
        h = science["scaling"]["holdout"]
        lines += [f"Held-out size: {h['parameters']} parameters. Predicted loss: {metric(h['predicted_loss'])}; measured loss: {metric(h['actual_loss'])}; absolute error: {metric(h['absolute_error'])}; relative error: {metric(100*h['relative_error'],2)}%.", ""]
        if science["identical_tokens"]:
            lines.append("**Null cleaning intervention:** the raw and clean token streams have identical SHA256 checksums. The extra cleaning rules made no change to the selected prefiltered TinyStories sample. Identical paired training results do not demonstrate a benefit of cleaning, and they must not be presented as evidence that cleaning improves model quality.")
        lines += ["", "Only four fitted scales are available for three curve parameters. Fixed token counts do not mean fixed FLOPs or compute-optimal convergence. Seed-bootstrap prediction intervals describe seed variability, not model misspecification. This study does not reproduce a raw Common Crawl cleaning experiment."]
    else:
        lines.append("Scaling results are not available in the configured directory.")
    lines += ["", "## 5. Keyword-conditioned post-training", "",
              "The existing pretrained model is reused. SFT masks prompt and padding targets and learns only response tokens. DPO and RLVR branch from the same final SFT checkpoint, which is also the frozen reference. Preference pairs are automatically generated by replacing one required word; they are synthetic constraint preferences, not human judgments.", "",
              "RLVR uses one on-policy grouped REINFORCE update per sampled group plus categorical next-token KL on sampled prefixes. It is not a full PPO or GRPO implementation. Reward is 0.75 times keyword coverage plus 0.15 for meeting minimum length and 0.10 for sufficient unique-word ratio. Equal-reward groups have zero relative advantage and are recorded. Sampling is untempered, untruncated categorical sampling with PAD/BOS excluded consistently from sampling and scoring.", "",
              "Train/dev/test tasks are grouped by the unordered keyword pair, so post-training test combinations are not used by the optimizer. Test stories originate from the original validation split; they are not a newly collected pretraining-blind dataset. All methods use the same evaluation seed per task. Training costs and sampling budgets differ across methods; the comparison is not compute-matched.", ""]
    if alignment:
        lines += ["| Model | Verified success | Keyword coverage | Repetition proxy | Mean words |", "|---|---:|---:|---:|---:|"]
        for row in alignment["evaluation"]:
            lines.append(f"| {row['model']} | {metric(row['success_rate'])} | {metric(row['keyword_coverage'])} | {metric(row['repetition'])} | {metric(row['mean_words'],1)} |")
        lines += ["", "Success means both words occur as whole words, the minimum word count is met, and the unique-word ratio reaches the configured threshold. These are proxies; manual reading is needed for grammar, coherence, repetition, and reward hacking.", "", "### Selected test outputs", ""]
        for name, samples in alignment["samples"].items():
            if samples:
                sample = samples[0]
                lines += [f"**{name}; required words: {', '.join(sample['words'])}**", "", sample["text"], ""]
    else:
        lines.append("Post-training has not produced a complete evaluation report yet. No success rate is claimed.")
    lines += ["", "## 6. Limitations and completion scope", "",
              "This submission covers parts 1–4 at the documented scale. Long-horizon Agent training and the self-judge module are omitted. Multi-GPU measurements and a real raw-web cleaning comparison are not included. Post-training is a narrow story constraint task, not general instruction alignment. Only one post-training training seed and a small test set are used; improvements must be judged from actual outputs and cannot be assumed.", "",
              "## 7. Reproduction and demonstration", "", "```bash", "python main.py experiment=full stat=train", "python main.py stat=attention attention=train", "python main.py stat=science", "python main.py stat=align", "python main.py stat=web", "```", "",
              "These commands describe the experiments; do not rerun completed training solely to regenerate this report. The demonstration page reads existing checkpoints and experimental files. It does not use a remote language-model API.", "",
              "## Figures", ""]
    chosen = ("training", "attention_bfloat16_4_forward_backward", "scaling_law", "data_quality", "alignment")
    for key in chosen:
        path = service.assets().get(key)
        if path and path.is_file():
            target = figures / f"{key}.png"
            shutil.copyfile(path, target)
            lines += [f"![{key}](figures/{key}.png)", ""]
    lines += ["## References", "", "- Eldan and Li. TinyStories. https://arxiv.org/abs/2305.07759", "- Kaplan et al. Scaling Laws for Neural Language Models. https://arxiv.org/abs/2001.08361", "- Hoffmann et al. Training Compute-Optimal Large Language Models. https://arxiv.org/abs/2203.15556", "- Rafailov et al. Direct Preference Optimization. https://arxiv.org/abs/2305.18290", "- Triton Fused Attention tutorial. https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html", ""]
    path = root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return dict(report=str(path), output_dir=str(root), full_results_available=bool(train.get("available") and attention and science and alignment),
                formatting="English Markdown draft; final NeurIPS typesetting remains required")

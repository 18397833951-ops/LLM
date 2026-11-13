"""独立基准。FP32 oracle 按 head 计算，避免长序列检查占满显存。"""

import math

import torch
from torch.nn import functional as F


def naive_attention(q, k, v, causal=True):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return probabilities @ v


@torch.no_grad()
def reference_attention(q, k, v, causal=True):
    output = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            output[batch, head] = naive_attention(
                q[batch, head].float(), k[batch, head].float(),
                v[batch, head].float(), causal)
    return output


def error_metrics(actual, expected, atol, rtol):
    actual, expected = actual.float(), expected.float()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
    return {
        "passed": finite and bool((difference <= atol + rtol * expected.abs()).all()),
        "max_abs_error": float(difference.max()) if finite else None,
        "rms_error": float(difference.square().mean().sqrt()) if finite else None,
        "atol": atol, "rtol": rtol,
    }

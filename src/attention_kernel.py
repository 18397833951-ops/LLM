"""分块 + online softmax 的 Triton 前向 Attention。

输入为连续的 [batch, heads, length, head_dim]，不实现 dropout 或反向。
算法参考 Triton 官方 Fused Attention 教程，采用普通指针以兼容 RTX 4090。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
                       SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr):
    query_block = tl.program_id(0)
    head = tl.program_id(1)
    rows = query_block * BM + tl.arange(0, BM)
    columns = tl.arange(0, BN)
    dims = tl.arange(0, D)
    base = head * N * D
    q = tl.load(Q + base + rows[:, None] * D + dims[None, :],
                mask=rows[:, None] < N, other=0)
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (query_block + 1) * BM)

    for start in range(0, end, BN):
        keys = start + columns
        k = tl.load(K + base + dims[:, None] + keys[None, :] * D,
                    mask=keys[None, :] < N, other=0)
        scores = tl.dot(q, k) * SCALE
        valid = keys[None, :] < N
        if CAUSAL:
            valid = valid & (rows[:, None] >= keys[None, :])
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        v = tl.load(V + base + keys[:, None] * D + dims[None, :],
                    mask=keys[:, None] < N, other=0)
        # 每加入一块 K/V，都将先前的分子、分母换算到新的最大值。
        accumulator = accumulator * alpha[:, None]
        accumulator += tl.dot(probabilities.to(q.dtype), v)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(O + base + rows[:, None] * D + dims[None, :], output,
             mask=rows[:, None] < N)


def triton_attention(q, k, v, *, causal=True, block_m=64, block_n=64, num_warps=4):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) < 1:
        raise ValueError("Q/K/V 必须同形且为非空 [batch, heads, length, head_dim]")
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous()
           for t in (q, k, v)):
        raise ValueError("Q/K/V 必须同设备、同精度且连续")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton Attention 需要 CUDA 上的 FP16/BF16 张量")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("当前只实现前向，请在 torch.no_grad() 下使用；不能用于训练")
    batch, heads, length, dim = q.shape
    if dim not in (32, 64, 128):
        raise ValueError("head_dim 只支持 32、64、128")
    if block_m not in (32, 64, 128) or block_n not in (32, 64, 128):
        raise ValueError("block_m/block_n 只支持 32、64、128")
    if num_warps not in (4, 8):
        raise ValueError("num_warps 只支持 4、8")
    output = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention_forward[(triton.cdiv(length, block_m), batch * heads)](
            q, k, v, output, length, dim, dim ** -0.5 * 1.4426950408889634,
            causal, block_m, block_n, num_warps=num_warps, num_stages=2)
    return output

"""分块 + online softmax 的 Triton Attention，含一阶反向。

输入为连续的 [batch, heads, length, head_dim]，不实现 dropout。
算法参考 Triton 官方 Fused Attention 教程，采用普通指针以兼容 RTX 4090。
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _attention_forward(Q, K, V, O, LSE, N: tl.constexpr, D: tl.constexpr,
                       SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, SAVE_LSE: tl.constexpr):
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
    if SAVE_LSE:
        tl.store(LSE + head * N + rows, running_max + tl.log2(running_sum), mask=rows < N)
    tl.store(O + base + rows[:, None] * D + dims[None, :], output,
             mask=rows[:, None] < N)


@triton.jit
def _backward_delta(O, DO, DELTA, N: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    dims = tl.arange(0, D)
    offsets = head * N * D + rows[:, None] * D + dims[None, :]
    output = tl.load(O + offsets, mask=rows[:, None] < N, other=0).to(tl.float32)
    grad = tl.load(DO + offsets, mask=rows[:, None] < N, other=0).to(tl.float32)
    tl.store(DELTA + head * N + rows, tl.sum(output * grad, axis=1), mask=rows < N)


@triton.jit
def _backward_dq(Q, K, V, DO, LSE, DELTA, DQ,
                 N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                 CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    base = head * N * D
    offsets = base + rows[:, None] * D + dims[None, :]
    q = tl.load(Q + offsets, mask=rows[:, None] < N, other=0)
    do = tl.load(DO + offsets, mask=rows[:, None] < N, other=0)
    lse = tl.load(LSE + head * N + rows, mask=rows < N, other=0)
    delta = tl.load(DELTA + head * N + rows, mask=rows < N, other=0)
    dq = tl.zeros((BM, D), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (block + 1) * BM)
    for start in range(0, end, BN):
        keys = start + tl.arange(0, BN)
        k = tl.load(K + base + keys[:, None] * D + dims[None, :],
                    mask=keys[:, None] < N, other=0)
        v = tl.load(V + base + keys[:, None] * D + dims[None, :],
                    mask=keys[:, None] < N, other=0)
        scores = tl.dot(q, tl.trans(k)) * (SCALE * 1.4426950408889634)
        valid = (rows[:, None] < N) & (keys[None, :] < N)
        if CAUSAL:
            valid = valid & (rows[:, None] >= keys[None, :])
        p = tl.exp2(tl.where(valid, scores - lse[:, None], -float("inf")))
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
        dq += tl.dot(ds, k)
    tl.store(DQ + offsets, dq, mask=rows[:, None] < N)


@triton.jit
def _backward_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV,
                   N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                   CAUSAL: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    keys = block * BN + tl.arange(0, BN)
    dims = tl.arange(0, D)
    base = head * N * D
    offsets = base + keys[:, None] * D + dims[None, :]
    k = tl.load(K + offsets, mask=keys[:, None] < N, other=0)
    v = tl.load(V + offsets, mask=keys[:, None] < N, other=0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    begin = 0
    if CAUSAL:
        begin = (block * BN // BM) * BM
    # 每个 program 独占一个 K/V 块，无 atomic_add 和跨 program 写冲突。
    for start in range(begin, N, BM):
        rows = start + tl.arange(0, BM)
        q = tl.load(Q + base + rows[:, None] * D + dims[None, :],
                    mask=rows[:, None] < N, other=0)
        do = tl.load(DO + base + rows[:, None] * D + dims[None, :],
                     mask=rows[:, None] < N, other=0)
        lse = tl.load(LSE + head * N + rows, mask=rows < N, other=0)
        delta = tl.load(DELTA + head * N + rows, mask=rows < N, other=0)
        scores = tl.dot(k, tl.trans(q)) * (SCALE * 1.4426950408889634)
        valid = (keys[:, None] < N) & (rows[None, :] < N)
        if CAUSAL:
            valid = valid & (keys[:, None] <= rows[None, :])
        p = tl.exp2(tl.where(valid, scores - lse[None, :], -float("inf")))
        dv += tl.dot(p.to(q.dtype), do)
        dp = tl.dot(v, tl.trans(do))
        ds = (p * (dp - delta[None, :]) * SCALE).to(q.dtype)
        dk += tl.dot(ds, q)
    tl.store(DK + offsets, dk, mask=keys[:, None] < N)
    tl.store(DV + offsets, dv, mask=keys[:, None] < N)


def _launch_forward(q, k, v, causal, block_m, block_n, num_warps, save_lse):
    batch, heads, length, dim = q.shape
    output = torch.empty_like(q)
    lse = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32) if save_lse else None
    with torch.cuda.device(q.device):
        _attention_forward[(triton.cdiv(length, block_m), batch * heads)](
            q, k, v, output, lse if save_lse else output, length, dim,
            dim ** -0.5 * 1.4426950408889634, causal, block_m, block_n, save_lse,
            num_warps=num_warps, num_stages=2)
    return output, lse


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, block_m, block_n, num_warps):
        output, lse = _launch_forward(q, k, v, causal, block_m, block_n, num_warps, True)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.settings = (causal, block_m, block_n, num_warps)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        causal, bm, bn, warps = ctx.settings
        batch, heads, length, dim = q.shape
        do = grad_output.contiguous()
        delta = torch.empty_like(lse)
        dq, dk, dv = (torch.empty_like(q) for _ in range(3))
        with torch.cuda.device(q.device):
            _backward_delta[(triton.cdiv(length, bm), batch * heads)](
                output, do, delta, length, dim, bm, num_warps=warps)
            _backward_dq[(triton.cdiv(length, bm), batch * heads)](
                q, k, v, do, lse, delta, dq, length, dim, dim ** -0.5, causal, bm, bn,
                num_warps=warps, num_stages=2)
            _backward_dkdv[(triton.cdiv(length, bn), batch * heads)](
                q, k, v, do, lse, delta, dk, dv, length, dim, dim ** -0.5, causal, bm, bn,
                num_warps=warps, num_stages=2)
        return dq, dk, dv, None, None, None, None


def triton_attention(q, k, v, *, causal=True, block_m=64, block_n=64, num_warps=4):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) < 1:
        raise ValueError("Q/K/V 必须同形且为非空 [batch, heads, length, head_dim]")
    if any(t.device != q.device or t.dtype != q.dtype or not t.is_contiguous()
           for t in (q, k, v)):
        raise ValueError("Q/K/V 必须同设备、同精度且连续")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton Attention 需要 CUDA 上的 FP16/BF16 张量")
    batch, heads, length, dim = q.shape
    if dim not in (32, 64, 128):
        raise ValueError("head_dim 只支持 32、64、128")
    if block_m not in (32, 64, 128) or block_n not in (32, 64, 128):
        raise ValueError("block_m/block_n 只支持 32、64、128")
    if num_warps not in (4, 8):
        raise ValueError("num_warps 只支持 4、8")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _Attention.apply(q, k, v, causal, block_m, block_n, num_warps)
    return _launch_forward(q, k, v, causal, block_m, block_n, num_warps, False)[0]

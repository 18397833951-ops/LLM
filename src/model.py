"""手写 decoder-only Transformer：Pre-LN、因果注意力、共享词嵌入。"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [item.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for item in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        # 用 FP32 计算 softmax，再转回混合精度，避免低精度归一化不稳定。
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        output = self.dropout(weights) @ v
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.proj(output))


class Block(nn.Module):
    def __init__(self, width, heads, context, dropout, ffn_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, context, dropout)
        self.ln2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_mult * width), nn.GELU(),
                                 nn.Linear(ffn_mult * width, width), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, layers, width, heads, context, dropout=0.1, ffn_mult=4):
        super().__init__()
        for name, value in dict(vocab_size=vocab_size, layers=layers, width=width,
                                heads=heads, context=context, ffn_mult=ffn_mult).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if width % heads:
            raise ValueError("width 必须能被 heads 整除")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(width, heads, context, dropout, ffn_mult)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError(f"输入应为 [batch, length]，length 在 1..{self.context} 内")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape != ids.shape:
                raise ValueError("targets 形状必须与输入相同，并已经右移一位")
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

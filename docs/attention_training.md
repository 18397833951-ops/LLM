# Attention 反向传播与训练路径测速

## 服务器一次运行

沿用已有 Linux RTX 4090 / PyTorch 2.8 / Triton 3.4 环境，务必激活 trackglimpse。无需新依赖、数据集或权重。

```bash
conda activate trackglimpse
cd ~/LLM
git pull --ff-only
python main.py stat=attention attention=train
```

36 个配置（16 个边界配置 + 20 个性能配置）分别比较三种实现的输出、dQ、dK、dV，共 432 项检查。全部通过才测速；失败保存报告并停止，不自动放宽误差容限。

测速依旧为 B=1/4、H=12、D=64，N=256/512/1024/2048/4096，FP16/BF16，共 20 个配置。分别测试朴素 PyTorch、强制 Flash SDPA、手写 Triton 的 forward 和 forward_backward，共 120 行结果。覆盖因果遮罩和非整块尾部；可用 attention.causal=false 切换整组非因果实验。

## 实现与数学

核心代码在 `src/attention_kernel.py`。前向为 online softmax，带梯度时额外保存每行 log2-sum-exp。无梯度调用沿用仅前向路径，不分配此缓存。通过 `torch.autograd.Function` 与自动求导连接，只支持一阶反向。

令 S=QKᵀ/sqrt(d)，P=softmax(S)，O=PV，上游梯度为 dO：

```text
dV = Pᵀ dO
dP = dO Vᵀ
delta_i = sum_j(P_ij * dP_ij) = sum_d(O_id * dO_id)
dS = P * (dP - delta)
dQ = dS K / sqrt(d)
dK = dSᵀ Q / sqrt(d)
```

反向分块重算 P，分别启动 delta、dQ、dK/dV kernel。每个 program 独占输出块，无全局 atomic_add，无 N×N 中间存储。exp2 使用 log2 分数还原概率；梯度按自然分数缩放 1/sqrt(d)，不额外乘 log2(e)。支持非连续上游梯度，输入仍要求连续 [B,H,N,D]。只需要部分输入梯度时仍计算三组梯度，不改变正确性。

## 检查与测量口径

FP32 oracle 按 head 单独建立 PyTorch autograd 图，关闭 TF32，不使用 SDPA 或手写 Triton 作为答案。采用同一个随机上游梯度比较 dQ/dK/dV。输出容限保持 FP16 atol=0.002/rtol=0.01、BF16 0.02/0.05；梯度独立为 FP16 0.004/0.02、BF16 0.035/0.05。按 `abs(error) <= atol + rtol*abs(reference)` 检查，并记录最大绝对误差和 RMS，未声称所有精度均达到 1e-3。

FP16/BF16 的低精度概率、dS 乘法及已舍入输出会产生近似误差，实际误差必须看服务器报告。单元测试还覆盖非因果、D=32、非连续上游梯度、重复反向及不同 tile 大小；本地无 CUDA 时 GPU 测试跳过。

forward 阶段关闭自动求导。forward_backward 阶段每次重新前向建图，再使用 autograd.grad 计算三个输入梯度，不累积 leaf.grad、不保留旧图。指标包含建图前向与反向，不包含 loss、优化器或完整 Transformer，不能直接作为端到端语言模型训练速度。每个阶段只与同阶段的朴素 PyTorch 比较加速比。

沿用 do_bench：预热 25ms、重复 100ms，排除首次 JIT，CUDA event 记录中位数及 20/80 分位数。新增峰值显存包含调用产生的输出、保存张量、梯度和临时张量；已有输入/上游梯度、allocator reserved 和其他进程不计入。较大的朴素反向需要较多显存，其他任务占用过多时会明确报 OOM，报告保留失败状态，不能视作完成。

## 输出与交付范围

新目录 `artifacts/attention_train/`：

- results.json：配置、环境、output/dQ/dK/dV 误差、状态和性能数据。
- benchmarks.csv：120 行，phase 区分 forward 和 forward_backward。
- 4 张前向图，另有 4 张带 `_forward_backward` 后缀的训练路径图。

旧命令 `python main.py stat=attention` 继续只测前向、保存到 artifacts/attention。新配置不覆盖之前结果；重复运行同一新配置会更新新目录中的报告。

当前本地为 Windows CPU 环境，梯度基准、配置和图表已检查；新增 Triton 反向的 GPU 编译、数值和性能须以服务器实测为准。尚未接入语言模型、实现 dropout、KV cache、二阶梯度或多卡实测。无需再进行 7 小时的全量训练。

算法参考 [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，自动求导接口参考 [PyTorch extending autograd](https://docs.pytorch.org/docs/2.8/notes/extending.html)。采用普通指针和固定分块，反向按上述公式实现，未使用 TMA 或 autotune。

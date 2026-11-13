# 手写 Triton Attention 前向实验

本文记录已完成的单卡前向实验。新增反向实现、梯度检查和训练路径测速见 [Attention 训练路径实验](attention_training.md)。

本阶段实现 `softmax(QKᵀ / sqrt(d) + causal_mask) V`，输入为连续的 `[B,H,N,D]` FP16/BF16 张量。支持 D=32/64/128、任意正整数 N（含非整块尾部）、因果或非因果模式。实现位于 `src/attention_kernel.py`，使用 64×64 分块、FP32 累加及 online softmax；不保存完整的 N×N 分数矩阵。

## 服务器运行

使用现有 Linux / RTX 4090 / PyTorch 2.8 CUDA 环境，无需重新下载数据或权重。

```bash
conda activate trackglimpse
cd ~/LLM
git pull --ff-only
python -m pip install -r requirements-attention.txt
python main.py stat=attention
```

PyTorch 2.8 Linux CUDA 安装通常已有匹配的 Triton 3.4。独立依赖文件不会安装/替换 PyTorch；Windows 的普通数据准备环境不需要安装 Triton。首次运行编译会比后续运行慢，需要服务器有可写的 Triton 缓存目录。整个任务不访问 Hugging Face。

一次命令按顺序执行：

1. 正确性检查：单 token、63/65/129 的非整块长度、D=64/128；以及每个正式性能配置。所有实现均与独立 FP32 逐 head 基准比较，关闭 TF32。错误或编译失败立即停止，保存失败信息，不输出虚假的性能结论。
2. 全部测速：B=1/4，H=12，D=64，N=256/512/1024/2048/4096，FP16/BF16，共 20 个配置，每个比较 3 种实现。因果模式、dropout=0。
3. 自动导出 JSON、CSV 和图表。仅显示进度条和阶段日志。

## 三种实现与测量口径

- `pytorch_naive`：直接计算完整分数矩阵，FP32 softmax 后转回输入精度，与第一部分模型的计算方式相同，无 dropout。
- `pytorch_flash_sdpa`：强制 `SDPBackend.FLASH_ATTENTION`，不允许回退到 math。环境不支持时任务明确报错。
- `triton`：本项目的手写分块 kernel。本命令只测前向；带梯度使用时会调用新增的一阶反向实现。

正确性容限为 `abs(actual-reference) <= atol + rtol*abs(reference)`，FP16 默认 atol=0.002/rtol=0.01，BF16 为 0.02/0.05；同时记录最大绝对误差和 RMS 误差，不声称两种精度均达到 1e-3。

先单独执行一次以排除 JIT，然后使用 `triton.testing.do_bench` 做 CUDA event 测量，预热 25ms、重复 100ms，记录中位数及 20/80 分位数。返回张量分配计入调用。显存指标为单次调用相对于已有输入的 **PyTorch allocated 新增峰值**，含输出，不是 nvidia-smi 的整卡显存，不含 allocator reserved、其他进程或编译开销。建议 GPU 上其他训练结束后再运行。

结果在 `artifacts/attention/`：

- `results.json`：环境、配置、全部检查误差、耗时和运行状态。
- `benchmarks.csv`：60 行性能数据，含相对于朴素 PyTorch 的加速比。
- `attention_float16_b1.png` 等 4 张图：耗时、额外峰值显存、加速比。

同一输出目录再次运行会更新报告；需要保留旧实验时指定不同的 `attention.output_dir`。进程中断会保留此前完成的数据；只有 `status=completed` 表示全部完成，CSV 中间结果不能当作完整结论。

## 结果分析与范围

分块消除了完整分数矩阵带来的二次中间存储，但精确 Attention 的计算量仍随 N² 增长，不能宣称耗时必然线性。与高度优化的 Flash SDPA 相比，手写实现可能更慢，应如实报告。

此命令交付单卡 **前向** 实验流水线，已在服务器完成 108 项比较和 60 行测速。新增反向代码和自动检查已加入独立 train 配置，GPU 结果待服务器确认。没有实现 dropout、KV cache、语言模型接入或多卡性能实测；原作业完整的多卡部分仍需补充。本实验不修改已经训练的模型，也不需要再次全量训练。

算法参考 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，接口与计时参考 [PyTorch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)、[SDPA 后端选择](https://docs.pytorch.org/docs/2.8/generated/torch.nn.attention.sdpa_kernel.html) 与 [Triton do_bench](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)。本项目采用普通指针和固定分块，未使用教程的 TMA、autotune 或反向实现。

# Transformer、测速与训练

先按 README 执行 `stat=benchmark`。本文按需查看，不要求逐条执行示例。

## 模型与环境

默认 `model=gpt100m`：14 层、隐藏维度 768、12 个头、上下文 256、FFN 扩展倍数 4、dropout 0.1。词表读取本地 BPE 的实际大小；8192 词表时有 **105,719,808 参数，约 0.106B**。

结构包含可学习位置编码、Pre-LN、共享输入输出 embedding、残差、GELU FFN，以及手写 Q/K/V、缩放点积、因果遮罩和 softmax。没有调用 `nn.Transformer`、现成 GPT 或 SDPA 代替核心 attention。

需要 PyTorch >= 2.4；本地 CPU 测试使用 2.6，CUDA/AMP 与显存需要服务器验证。沿用已可用的服务器环境。`requirements.txt` 只添加数据、配置和绘图依赖，不替换 PyTorch。

若环境缺少合适版本，按 [PyTorch 官方历史安装说明](https://docs.pytorch.org/get-started/previous-versions/)选择对应 wheel。驱动支持 CUDA 12.4 的机器可考虑官方 2.6/cu124；先确认环境用途，避免影响其他项目。

## 编码与采样

首次测速或训练会自动将本地故事编码为 token 流，每篇末尾加 EOS，不联网。缓存位于 `artifacts/tokens/`，按数据与词表文件 SHA256 区分，文件损坏会报错。

默认按 epoch 依次遍历本地训练 token 流，标签比输入向后错一位；窗口可跨故事边界，由 EOS 标记，不另外重置 attention。每遍的有效目标 token 恰好覆盖一次，尾批补齐位置的标签设为 -100，不计入 loss，梯度按实际有效 token 数加权。每遍顺序固定，原始文档顺序由数据准备时的 shuffle 决定。显式设置 train.max_steps 时使用兼容的随机窗口模式，不保证完整遍历。

验证按固定顺序遍历，默认全部验证目标 token，包括不足完整窗口的尾部。loss 按有效目标 token 数加权，perplexity 为相同词表下 loss 的指数。`eval.max_tokens=null` 可评估全部验证数据。

## 测速

```bash
python main.py stat=benchmark
```

默认 micro-batch 为 8、梯度累积 4、上下文 256，每个优化步骤 8192 个目标 token。先预热 5 步，再测 20 步，CUDA 计时前后同步。包含 batch 读取、前后向、梯度裁剪和优化更新，不包含首次编码、验证、生成和 checkpoint 保存；不保存测速模型权重。

相较之前的 4×4，每次更新的训练量翻倍，不表示速度翻倍；比较速度应使用相同 token 预算下的 tokens_per_second。固定 1000 步时，处理的数据量和实际耗时都会改变。

auto 精度在 CUDA 上优先 BF16，否则 FP16；CPU 用 FP32。FP16 配套 GradScaler，非有限 loss 或梯度停止更新。默认配置测试的是正式模型，测试代码内部的小尺寸仅用于正确性验证。

主要输出：`parameters`、`tokens_per_second`、`seconds_per_step`、`peak_allocated_gib`、当前训练预算的 `estimated_training_hours`、`benchmark.estimate_tokens` 预算的 `estimate_budget_hours`。GPU 被其他任务占用时实测速度会变化。

显存不足时先降低 `train.batch_size`，必要时增大 `train.grad_accum` 保持每步 token 数，再重新测速。

## 正式训练

确认预算后执行：

```bash
python main.py stat=train
```

默认 train.epochs=3、train.max_steps=null，由实际编码 token 数自动计算每个 epoch 和总更新步数。8×4、context=256 时每步最多 8192 个有效目标 token；每遍末尾最后一步可能少于此数。训练开始会输出 training_plan，日志和 checkpoint 记录实际 seen_tokens。

当前本地若仍只有 1 万篇故事，跑 3 个 epoch 也只是把这 1 万篇学习三遍，不会自动下载全量。全量实验须准备对应全量数据与词表，并使用 experiment=full，详见 [全量训练流程](full_run.md)。

优化使用 AdamW、warmup、cosine 衰减、梯度累积、梯度裁剪和可选混合精度。保存训练初始、定期及结束时的固定提示词续写，便于对比。

默认训练目录 `artifacts/gpt100m_epochs/`；全量配置使用 `artifacts/gpt100m_full/`：

| 文件 | 含义 |
|---|---|
| config.json / model_info.json | 运行配置、参数量和编码来源 |
| metrics.jsonl | 每步 loss、LR、吞吐和定期验证 |
| samples.jsonl | 不同阶段的固定提示词续写 |
| last.pt | 最近保存的完整训练状态 |
| best.pt | 已评估 checkpoint 中验证 loss 最低的状态 |

完整 checkpoint 包含模型、优化器、GradScaler、采样 RNG、CPU/CUDA RNG 和步数；学习率由保存的计划参数与步数重算。每个含优化器的 checkpoint 约 GB 级，需为 best/last 和临时写入留足磁盘空间。

默认每 5000 步验证、保存、生成，每个 epoch 结束和整次训练结束时也执行。意外中断只能从最近一次保存恢复，期间未保存步骤会丢失。输出目录存在且未设置 resume 时不会覆盖。

## 断点恢复

训练与测速默认有进度条，相关选项在 `configs/config.yaml` 的 `progress` 中。`progress.enabled=false` 可以关闭，`progress.min_interval` 控制刷新间隔（默认 0.5 秒）。终端日志与进度条协调输出，Hydra 日志和 metrics.jsonl 仍正常保存。进度同时显示当前 epoch 和累计优化步数，恢复时从 checkpoint 对应 epoch/窗口继续；ETA 随实际执行时间估计，验证与保存可能使其波动。使用 stop_after 提前结束时，百分比仍相对于完整 max_steps 计划。

```bash
python main.py stat=train train.resume=artifacts/gpt100m_epochs/last.pt
```

必须保持模型、数据、词表、优化参数、训练计划（epochs 或 max_steps）及精度一致，并用同一 output_dir。`max_steps` 是整次计划目标，不是在已有步骤上再加。GPU/驱动/后端变化时不保证浮点结果逐位相同。

旧 4×4、1000 步实验恢复时需显式设置 train.batch_size=4、train.max_steps=1000、train.output_dir=artifacts/gpt100m，不能直接使用新的 epoch 模式续训旧的随机窗口实验。新实验使用独立目录保留旧结果。

可选的续训验证：第一次运行设置 `train.stop_after=20` 提前停下并保存，恢复时取消该限制。原始学习率计划仍由 epochs 计算出的总步数（或显式 max_steps）决定，不会按 stop_after 截短。

扩大训练预算请在首次正式运行前设置 train.epochs（或兼容模式的 train.max_steps）。更改已有计划需另立实验，不声称与原计划连续训练等价。

## 评估、生成与曲线

训练成功保存 checkpoint 后，按需要执行：

```bash
python main.py stat=eval
python main.py stat=generate
python main.py stat=plot
```

评估和生成默认读 `best.pt`。模型结构取自 checkpoint，词表 SHA256 必须一致；生成不需要下载模型，评估还要校验数据来源。

生成是故事续写，不是指令问答。默认温度 0.8、top-k 50，`generate.temperature=0` 使用贪心选择。上下文超长时截取最后一个窗口，无 KV cache。

`plot` 生成 `loss_curve.png`、`metrics.csv`。中断恢复可能重放部分日志，同一步保留最后一条，不以日志行数代替步数。

## 验证

测试覆盖循环 attention 数值对照、因果遮罩、标签位移、共享权重、有限梯度、小批次过拟合、token 流尾部评估、生成不改变训练 RNG，以及中断恢复参数与连续训练一致。

```bash
python -m unittest discover -s tests -v
```

核心实现：`src/model.py`、`src/token_data.py`、`src/training.py`。配置：`configs/model/gpt100m.yaml`、`configs/train/default.yaml`。

当前只完成代码与本地 CPU 验证，没有在本地声称完成服务器 GPU 训练；正式效果与耗时应依据服务器结果判断。

## 手动停止

Ctrl+C 按 Python 默认行为直接终止，不自定义拦截、不额外保存当前步。续训从最近的 last.pt 开始，未保存的更新需要重跑。若正在保存时中断，原 last.pt 保留，未写完的临时文件不会代替它；强制断电、磁盘故障不在此保证内。

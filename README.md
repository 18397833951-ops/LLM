# 从零训练英文语言模型

已实现手写 BPE、约 0.1B 的 Transformer、epoch 训练、测速、断点恢复、评估、生成和曲线导出。全量训练已在服务器完成：3 个 epoch，最佳验证 loss 0.8385，困惑度 2.313。

## 第二部分：Triton Attention

在服务器激活 `trackglimpse` 并更新代码后，一次跑完前向/梯度检查、前向及前向+反向测速和绘图：

```bash
python -m pip install -r requirements-attention.txt
python main.py stat=attention attention=train
```

结果保存到 `artifacts/attention_train/`，保留已完成的前向结果。前向与反向已在服务器通过，共 432 项比较、120 行测速。见 [Attention 训练路径实验](docs/attention_training.md)。

## 当前阶段：数据清洗与 Scaling Law

Attention 单卡前向/反向已在服务器完成 432 项比较和 120 行测速。第三部分代码已接入，默认复用服务器已有 TinyStories 做进一步清洗对照，并训练四档小模型与独立大模型检查外推：

```bash
python main.py stat=science
```

默认包含 18 次短训练，每次 2000 步、同样的 token 预算。自动恢复断点并跳过已完成实验，输出保存在 `artifacts/science/`；实际实验结果待服务器运行。范围、局限和说明见 [第三部分实验](docs/science.md)。

## 当前训练配置

- batch_size=8，grad_accum=4。
- 默认训练 3 个 epoch，每遍完整覆盖当前训练数据，包括尾部。
- 每 5000 步验证并保存，每个 epoch 结束时也验证、保存。
- best.pt 按验证 loss 选择，last.pt 用于恢复。
- 有进度条，显示当前 epoch、累计步数、预计剩余时间和 loss。
- Ctrl+C 正常终止；之后从最近保存的 last.pt 恢复，未保存部分重跑。

**加 epoch 不等于已经准备好全量数据。** 若仍使用之前的 1 万篇，训练三遍依然只是在学习这 1 万篇。

## 首次复现：准备全量数据

在本地 Windows 的 D:\LLM 目录更新代码后执行：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe main.py experiment=full stat=prepare
```

这次读取全部官方训练和验证 split，过滤重复后保存到 data/tinystories_full。需要本地能访问 Hugging Face；之前的小样本目录不会覆盖。完成后按 [全量训练流程](docs/full_run.md) 继续，未准备好对应数据和词表时不要直接启动全量训练。

## 文档

- [全量训练流程](docs/full_run.md)：按顺序完成本地准备、上传数据、服务器训练和恢复。
- [Attention 实验](docs/attention.md)
- [清洗与 Scaling Law](docs/science.md)
- [数据准备](docs/data_usage.md)
- [分词器](docs/tokenizer.md)
- [模型与训练说明](docs/training.md)

各文档的可选参数不需要逐条执行。显式设置 train.max_steps 可兼容旧随机窗口实验；默认保持 null 使用 epoch 模式。

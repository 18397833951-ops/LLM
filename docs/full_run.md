# 全量实验：按顺序操作

统一使用 `experiment=full`，这样数据、词表和权重都与原来的小样本实验分开。默认 8×4，3 个 epoch，每 5000 步及 epoch 结束时验证和保存，验证使用当前完整验证集。

## 1. 本地准备数据和词表

本地可以访问 Hugging Face。在 Windows PowerShell 的 `D:\LLM` 中依次执行，上一条成功才运行下一条：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe main.py experiment=full stat=prepare
.\.venv\Scripts\python.exe main.py experiment=full stat=inspect
.\.venv\Scripts\python.exe main.py experiment=full stat=tokenizer
.\.venv\Scripts\python.exe main.py experiment=full stat=tokenizer_check
```

全量读取会传输 GB 级数据，去重、词表学习和检查占用 CPU、内存和磁盘。进度中的总量为空时表示上游流尚未提供准确总数；实际完成量和来源仍记录到 manifest。完整读取不等于保留所有原始行，空文本、重复和验证集与训练集的重叠会被过滤。

这次从全量训练集重新学习词表，不能拿旧小样本模型继续训练；不同词表的编号含义可能改变。

## 2. 上传到服务器

先在服务器创建目标父目录：

```bash
mkdir -p ~/LLM/data ~/LLM/artifacts
```

再在本地 `D:\LLM` 运行：

```powershell
scp -r .\data\tinystories_full tjw@192.168.50.25:/home/tjw/LLM/data/
scp -r .\artifacts\tokenizer_full tjw@192.168.50.25:/home/tjw/LLM/artifacts/
```

不要漏掉两个目录中的 manifest、tokenizer.json 和来源报告。代码仍使用 GitHub 更新，不需要上传 .venv、work、缓存或旧权重。

## 3. 服务器检查、测速和训练

在服务器已可用的 Python 环境中执行：

```bash
cd ~/LLM
git pull --ff-only
python main.py experiment=full stat=inspect
python main.py experiment=full stat=tokenizer_check
python main.py experiment=full stat=benchmark
```

先确认数据检查与测速成功，再训练：

```bash
python main.py experiment=full stat=train
```

首次自动编码，随后会打印实际 targets_per_epoch、steps_per_epoch、total_steps。一个 epoch 覆盖 token 流的全部 next-token 目标，末批不足会补齐并在 loss 中忽略补齐位置。每遍顺序固定，不做有放回抽样。具体步数与耗时以实际编码统计为准。

输出目录：`artifacts/gpt100m_full/`。不会覆盖旧 `artifacts/gpt100m/` 或默认小样本 epoch 实验目录。

## 4. 中途停止和恢复

按 Ctrl+C 即正常终止，程序不额外拦截保存。每 5000 步和 epoch 末尾会保存 last.pt，未保存部分重跑即可。

```bash
python main.py experiment=full stat=train train.resume=artifacts/gpt100m_full/last.pt
```

保持 epochs、batch、优化器和数据/词表不变。checkpoint 记录更新步数、token 计数和训练计划，由它确定恢复的 epoch 与读取位置；不重新从头遍历已保存的部分。修改 epochs 会改变学习率计划，因此训练开始后不随意修改总 epoch 数。

## 5. 训练后查看效果

```bash
python main.py experiment=full stat=eval
python main.py experiment=full stat=generate
python main.py experiment=full stat=plot
```

生成和评估默认使用 best.pt。3 epoch 是训练上限，不保证每一遍都改善验证结果，选择最终模型看验证 loss 和固定提示词输出。

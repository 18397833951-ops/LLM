# 从零训练英文语言模型

已实现手写 BPE、约 0.1B 的 Transformer、epoch 训练、测速、断点恢复、评估、生成和曲线导出。全量训练已在服务器完成：3 个 epoch，最佳验证 loss 0.8385，困惑度 2.313。

## 第二部分：Triton Attention

在服务器激活 `trackglimpse` 并更新代码后，一次跑完前向/梯度检查、前向及前向+反向测速和绘图：

```bash
python -m pip install -r requirements-attention.txt
python main.py stat=attention attention=train
```

结果保存到 `artifacts/attention_train/`，保留已完成的前向结果。前向与反向已在服务器通过，共 432 项比较、120 行测速。见 [Attention 训练路径实验](docs/attention_training.md)。

## 第三部分：数据清洗与 Scaling Law

第三部分已在服务器完成 18 次等 token 预算实验，大模型外推相对误差约 2.46%。清洗前后的 token 校验值相同，说明本次规则没有改变已有 TinyStories 样本，不声称清洗提升效果。以下命令仅用于首次复现，已有结果无需重跑：

```bash
python main.py stat=science
```

输出保存在 `artifacts/science/`。范围与局限见 [第三部分实验](docs/science.md)。

## 当前阶段：第四部分与演示页面

复用已有故事模型与数据，不下载另一套模型。在服务器更新代码后执行：

```bash
python main.py stat=align
```

自动完成 SFT、从同一 SFT 权重分支的 DPO/RLVR、统一测试、比较图和英文报告草稿。偏好为自动构造的关键词约束偏好，RLVR 为 on-policy 分组 REINFORCE+KL；效果以服务器运行结果为准。第五部分 Agent 不在本次范围。

完成后启动本地演示服务：

```bash
python main.py stat=web
```

在 Windows 电脑用 SSH 转发 6006 端口后，浏览器打开 http://127.0.0.1:6006。详见 [第四部分说明](docs/alignment.md) 和 [页面使用说明](docs/demo.md)。

## 当前训练配置

英文问候是可选的独立微调分支：`python main.py stat=chat`。自动准备项目内小型合成对话数据、复用原权重做 SFT，再保存真实测试回复。完成后页面可选“英文问候 SFT”，保留有限的最近聊天上下文；不保证通用聊天能力。见 [英文问候说明](docs/chat.md)。

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
- [SFT / DPO / RLVR](docs/alignment.md)
- [演示页面](docs/demo.md)
- [英文问候扩展](docs/chat.md)
- [数据准备](docs/data_usage.md)
- [分词器](docs/tokenizer.md)
- [模型与训练说明](docs/training.md)

各文档的可选参数不需要逐条执行。显式设置 train.max_steps 可兼容旧随机窗口实验；默认保持 null 使用 epoch 模式。

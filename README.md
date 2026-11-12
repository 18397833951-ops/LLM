# 从零训练英文语言模型

已实现数据准备、手写 BPE、约 0.1B 的手写 Transformer，以及测速、训练、续训、评估、生成和曲线导出。正式 GPU 训练尚待在服务器执行。

## 当前步骤：在服务器测速

你已经完成数据和分词器准备，不需要重新下载或训练词表。在服务器的 LLM 项目根目录，激活原来的环境后依次执行：

```bash
git pull --ff-only
python -m pip install -r requirements.txt
python main.py stat=benchmark
```

模型任务需要已有 **PyTorch >= 2.4 的 CUDA 环境**，requirements.txt 不会替换你的 PyTorch。

测速直接使用约 0.1B 的模型，不要求先训练小模型。默认预热 5 步，再测 20 步，报告参数量、精度、每秒 token 数、显存及训练时长估算。结果保存在 runs/benchmark/ 下对应运行目录中，测速不保存模型权重。

编码、预热、测速和训练默认显示进度条，无需额外参数。训练进度条显示步数、百分比、预计剩余时间、loss、最佳验证 loss 和吞吐；验证、生成和保存时会显示当前阶段。

**先把测速输出发回来，再根据实际速度和显存确定正式训练预算。** 估算不包含首次编码、验证、生成和保存模型的时间。

如果出现显存不足，保留完整报错，不要连续重试原配置。

## 其他步骤说明

- [数据准备](docs/data_usage.md)：已准备成功不用重跑。
- [BPE 分词器](docs/tokenizer.md)：已有词表不用重训。
- [模型训练](docs/training.md)：结构、正式训练、恢复、评估、生成和可选参数。

这些文档按需查看，不需要逐条执行其中的示例。

数据来源：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。

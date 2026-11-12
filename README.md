# 从零训练英文语言模型

当前已实现 TinyStories 数据准备、检查和手写 BPE 分词器。Transformer 和模型训练将在后续实现。

## 运行方法

在服务器上进入 `LLM` 项目根目录，使用已激活的 Python 3.11 环境，按顺序执行下面三条命令。每条成功后再执行下一条。

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 准备数据
python main.py stat=prepare

# 3. 检查数据并查看故事
python main.py stat=inspect
```

默认准备 **10,000 篇训练故事和 1,000 篇验证故事**，保存到 `data/tinystories/`。

检查结束后，报告中 `passed` 为 `true`、`issues` 为 `[]`，表示本次数据检查通过；终端还会显示几篇故事供查看。

`stat=prepare` 首次准备时需要联网，只需成功执行一次。数据已准备好后，直接运行 `stat=inspect` 即可；再次执行 `prepare` 会提示目录已存在，不会覆盖原数据。

如果命令报错，先停止后续步骤，保留完整报错信息。

## 数据检查通过后：训练分词器

已完成数据准备的同学不用重新下载。在服务器项目根目录，依次执行：

```bash
# 学习并保存 BPE 词表，只使用训练集
python main.py stat=tokenizer

# 检查文字 → 编号 → 原文是否一致
python main.py stat=tokenizer_check
```

这两条命令只读本地数据、不联网，也不需要 GPU。检查结果 `passed: true`、`issues: []`，且两组 `roundtrip_mismatches: 0` 表示还原检查通过。它不代表语言模型已经训练完成。

词表保存在 `artifacts/tokenizer/tokenizer.json`，训练和检查摘要也在该目录。已经训练过词表时直接执行检查，训练命令不会覆盖已有词表。

算法说明和可选参数见 [分词器说明](docs/tokenizer.md)，首次按上面两条命令执行即可。

## 可选参考

调整数据量、环境配置及代码测试见 [数据准备进阶说明](docs/data_usage.md)。这些是可选参考，不需要逐条执行。

数据来源：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。

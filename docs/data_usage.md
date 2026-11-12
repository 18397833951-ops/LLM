# 数据准备进阶说明

本文是可选参考，不需要逐条执行。首次使用请按 [README](../README.md) 的三步操作；只有需要调整样本数量、排查问题或复现实验时，才查看下面对应的部分。

## 1. 服务器环境

本步骤不使用 GPU，不需要安装 PyTorch。建议使用独立 Python 3.11 环境，在项目根目录执行：

```bash
conda create -n llm python=3.11 -y
conda activate llm
python -m pip install -r requirements.txt
python -m pip check
```

依赖范围用于兼容；首次验证通过后，可执行 `python -m pip freeze > environment-lock.txt` 保存本机实际版本。不要在其他项目环境中直接修改依赖。

## 2. 准备数据

```bash
python main.py stat=prepare
```

默认使用官方训练集与验证集，分别保存 10,000 和 1,000 篇有效故事。数据以流式方式读取，不主动下载整个仓库，但依然需要网络，会传输部分分片、预取和 shuffle 缓冲数据；传输量不等于最终样本文件大小。

```bash
# 先用更小的数量跑通，指定单独目录
python main.py stat=prepare data.train_size=100 data.val_size=20 data.output_dir=data/debug

# 改变正式样本量
python main.py stat=prepare data.train_size=50000 data.val_size=2000 data.output_dir=data/stories_50k

# 完整遍历：耗时、磁盘和内存需求均会上升，不建议首次运行就用
python main.py stat=prepare data.train_size=null data.val_size=null data.output_dir=data/stories_full
```

输出示例：

```text
data/tinystories/
├── train.jsonl       # 每行一篇训练故事
├── val.jsonl         # 每行一篇验证故事
└── manifest.json     # 数据来源、commit、配置、版本、过滤计数、文件 SHA256
```

每行包含 `id`、`text`、`text_sha256`。原文（含换行）保留，JSONL 会转义文本中的换行。`id` 是本次采样顺序标识，不是上游永久文档 ID。

默认跳过空文本、过滤 split 内重复，并剔除验证样本中与已选训练样本重复的故事；不会把训练样本移入验证集。重复指纹使用 Unicode NFC 和合并空白后的 SHA256，不做近似语义去重。

两组全部成功后才发布输出目录。网络失败、字段错误、可用样本不足时返回错误，不发布不完整目录。程序不覆盖已有输出，重跑请换 `data.output_dir`；无需手动删除已有数据。

## 3. 检查和查看故事

```bash
python main.py stat=inspect
python main.py stat=inspect data.output_dir=data/debug
python main.py stat=inspect inspect.samples=5 inspect.preview_chars=500
```

检查不访问网络。报告包含数量、空文本、重复数量、训练/验证重复、文本字符长度、文件校验和故事预览，同时保存 `inspection.json`。检测到质量问题时命令返回非零退出码；JSON 损坏、文件缺失会直接报错。

数据检查以 `manifest.json` 为来源依据，不能仅通过修改当前 YAML 改写已有数据的来源。

## 4. Hydra 配置

```text
main.py
configs/config.yaml
configs/data/tinystories.yaml
src/data.py
tests/test_data.py
```

```bash
# 只查看合并后的配置，不下载数据
python main.py --cfg job

# 固定上一次使用的 commit（用 manifest 中实际 SHA 替换）
python main.py stat=prepare data.revision=COMMIT_SHA data.output_dir=data/reproduced
```

数据任务使用 `stat=prepare`、`stat=inspect`；分词器任务另见 [分词器说明](tokenizer.md)。不存在的任务会报错。主配置默认 `inspect`，直接运行不触发下载。相对路径基于启动目录，因此请始终在项目根目录运行。Hydra 不切换工作目录，每次执行的日志和配置保存在 `runs/`。

采样是固定 seed 的缓冲区 shuffle，不是对整个数据集做全局均匀随机抽样。固定 commit、配置和依赖版本有助于复现；manifest 同时保存实际输出文件校验值。

## 5. 本地离线验证

```bash
python -m unittest discover -s tests -v
```

测试用人工构造的几条文本验证去重、分区、数据损坏和失败清理，不读取或下载正式数据。它不能代替服务器对真实上游网络、权限、split 和流式读取的验证。

完整去重和检查会在内存中保存文本指纹，内存随样本数量增长；扩大到整个数据集时应先监控内存，必要时改用磁盘索引。

数据来源：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，遵循其数据许可。实现参考：[Datasets 流式读取](https://huggingface.co/docs/datasets/stream)、[Hydra 1.3](https://hydra.cc/docs/1.3/intro/)。

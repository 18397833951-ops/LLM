# 从零训练英文语言模型

当前完成的是第一步：准备和检查 TinyStories 数据。分词器和模型将在后续实现。

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

## 可选参考

正常使用只需上面的三步。调整数据量、环境配置、Hydra 参数及代码测试见 [数据准备进阶说明](docs/data_usage.md)，不需要逐条执行。

数据来源：[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)。

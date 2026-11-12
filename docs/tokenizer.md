# BPE 分词器说明

首次使用按 README 的两条分词器命令操作即可，下面是可选参考。

## 训练过程

代码在 `src/tokenizer.py`，使用 Python 标准库手写，未调用现成分词器训练库。

1. 将文本按字母、数字、空白、标点、下划线分块，保留所有字符。
2. 每个分块编码成 UTF-8 字节，建立 256 个基础 token。
3. 统计相邻 token 对的加权频次，将最常见的一对合并成新 token。
4. 重复合并，直到达到目标词表大小，或没有达到最小频次的 token 对。

重复分块只保留一份并记录权重。训练使用 token 对到分块的索引，只更新受影响的分块；最大堆使用延迟验证来选择最新频次。频次相同时按 token 对编号排序，保证确定性。

编码时按学习到的合并顺序应用规则；解码时将对应字节拼接后转换为 UTF-8 文本。未见过的字符仍可退回到基础字节，不需要 unknown token。BPE 不跨故事或预分块边界合并，因此与 GPT-2 等分词器的规则不完全相同。

## 词表与特殊 token

- `0..255`：原始字节。
- `256`：`<pad>`。
- `257`：`<bos>`。
- `258`：`<eos>`。
- `259` 起：学习到的合并 token。

默认目标词表 8192，包括以上特殊 token；若数据不足，实际词表可以更小，后续模型需读取实际 `vocab_size`。

输入中的字面字符串 `<eos>` 被视为普通文字。只有显式指定 `add_bos=True`、`add_eos=True` 才会添加特殊 token。默认解码跳过特殊 token；普通 `encode/decode` 严格保留空白、换行和 Unicode 内容，不做繁简转换或规范化。

任意模型生成的 token 序列可能拼出无效 UTF-8，生成阶段可显式使用 `decode(ids, errors="replace")`；还原检查使用严格解码，不能用替换字符掩盖错误。

## 本地使用

```python
from src.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load("artifacts/tokenizer/tokenizer.json")
text = "Once upon a time, there was a little cat."
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
```

## 可选参数

配置在 `configs/tokenizer/bpe.yaml`。下面都是可选示例，不需要全部运行。

```bash
python main.py stat=tokenizer tokenizer.vocab_size=4096 tokenizer.output_dir=artifacts/bpe4096
python main.py stat=tokenizer_check tokenizer.output_dir=artifacts/bpe4096
```

`train_documents: null` 表示使用当前本地全部训练样本；设置正整数可限制用于学习规则的故事数量。`check_documents: null` 表示检查当前本地全部训练、验证故事。修改输出目录后，检查时需使用同一目录。

`tokenizer.json` 保存格式版本、分块规则、特殊 token、按顺序排列的合并规则及训练来源。`train_report.json` 记录来源文件 SHA256、实际文档数、目标与实际词表大小；`check_report.json` 记录两组还原结果、字节/token 比和演示输入。

数据文件必须与数据准备时的 manifest 校验一致，检查任务还会验证词表的训练来源与当前训练数据一致。验证集不参与学习 BPE。字节/token 比用于描述本分词器的压缩情况，不直接等同于语言模型能力，也不能直接用于比较不同词表模型的 loss。

分词器训练是 CPU 任务。当前训练样本中唯一分块及其索引保存在内存中，扩大语料时需要关注内存占用。

## 正确性检查

```bash
python -m unittest discover -s tests -v
```

测试包括：增量统计与朴素重算参考一致、重叠对合并、Unicode 与空白还原、特殊 token、规则保存加载、未知 ID、损坏词表，以及训练任务不读取验证集。

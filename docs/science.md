# 第三部分：数据清洗对照与参数规模实验

## 服务器运行

使用现有的 `data/tinystories_full` 和 `artifacts/tokenizer_full`，不联网、不增加依赖。先激活原训练环境，更新代码后执行：

```bash
conda activate trackglimpse
cd ~/LLM
git pull --ff-only
python main.py stat=science
```

一条命令完成语料准备、共享词表编码、18 次短训练、配对对照、幂律拟合、独立外推验证和绘图。GPU 实测结果需在服务器运行后确认。

默认三个种子（42/43/44），每个种子运行：

1. 约 14M 模型学习进一步清洗前的语料。
2. 同结构同初始种子的约 14M 模型学习进一步清洗后的语料，同时复用为 scaling 的一档。
3. 约 2M、8M、30M 三档模型学习清洗后语料。
4. 约 0.106B 模型学习清洗后语料，仅用于检验外推，不参与拟合。

每次从零训练，预算为 16,384,000 个已处理目标 token；batch_size=8、grad_accum=4、context=256，共 2000 次更新。该预算独立于第一部分的 3 epoch 全量训练，既不加载那次权重，也不拿其 loss 混作等预算对照。语料为带 EOS 的流式 token；随机抽窗口、有放回，报告 sampled_epoch_equivalent 仅表示已处理 token/语料 token 数，不保证完整覆盖。

## 语料与清洗范围

默认从已有训练文件按顺序选取 50,000 条可用文本，官方验证文件前 2000 条作为共同验证集。首先校验源文件 SHA256 和词表来源，保持词表冻结。词表来自此前完整训练集，不读取验证集重新学习。所有官方验证文本均用于训练候选的重叠排除（精确匹配和 LSH 检出的近重复），两组遵循相同排除规则。

此前 TinyStories 文件已经做过基础长度过滤和精确去重，因此 raw 在此仅指“本次进一步清洗前”，不代表未经处理的 Common Crawl。报告会记录这一区别，不注入人造垃圾来制造清洗收益。第三部分原始网页 dump 的完整实证仍需使用下面的本地 WET 入口补充。

各过滤器可在 `configs/science/default.yaml` 中独立开关：

- HTML/script/style 清理与实体解码。
- 长度上下限。
- 英文语言启发式（ASCII 字母占全部字母比例），不冒充 fastText 语言分类器。
- 质量启发式：符号比例、独特单词比例、lorem ipsum。
- 邮箱、常见长数字/电话格式脱敏。不能覆盖所有 PII，规则可能误判。
- NFC/空白归一化后的精确去重。
- 5-word shingles、32 个 MinHash 签名、8 个 LSH band 检索，候选用真实 Jaccard>=0.8 确认。

MinHash/LSH 候选检索存在漏检概率；真实 Jaccard 复核避免仅凭签名碰撞删除文本。报告记录按流水线首次命中原因统计的删除数、脱敏命中数和保留数。数据包含 raw/clean/val 三份，均不上传 GitHub。

对照组使用同一源候选池，清洗组使用其过滤/变换结果；共用词表、验证集、模型、初始种子、优化器设置和处理 token 预算。清洗改变了语料组成和唯一文本数量，等 token 对照不隔离全部因果因素。结果可能没有改善甚至退化，应如实报告，不预设 clean loss 更低。

如需换成实际的本地原始网页语料，在服务器已有文件路径上设置：

```bash
python main.py stat=science science.source_path=/path/to/sample.wet.gz science.source_format=wet science.output_dir=artifacts/science_wet
```

也支持每行 `{"text":"..."}` 的本地 JSONL/.jsonl.gz。WET 按记录的 Content-Length 读取 conversion 正文，不将 WARC 头混入文本，异常截断立即报错。外部源仍使用 TinyStories 词表与验证集，因此是跨域训练对照，不能当成同分布网页语言建模评估；若研究网页域本身，应另准备独立网页验证集与仅训练语料学习的共享词表。本实现不自动下载 Common Crawl。

## Scaling Law 的含义和局限

使用四个小模型的真实参数量及各自三个种子的**固定预算末尾验证 loss 均值**，拟合：

```text
L(N) = floor + amplitude * (N / 1e6)^(-alpha)
```

alpha 在配置范围内网格搜索；给定 alpha 后拟合非负 amplitude 和 floor。拟合不依赖 SciPy。保存 RMSE、R²、边界警告、留一规模验证，以及对独立大模型的预测、实测和外推误差。目标大模型从四个拟合点中排除，不能先纳入再称作外推。

种子 bootstrap 对每个规模使用相同重采样种子索引，记录预测的 2.5/97.5 分位数；仅描述这三个种子的运行波动，不涵盖幂律假设错误，不作严格的总体置信保证。四个规模拟合三个参数仅一个残差自由度，留一规模只剩三个点，都是弱诊断。指数落在边界或实测不单调时报告提示，不隐藏失败结果。

这是固定 token 预算下的参数规模切片，不是固定 FLOPs、充分收敛或 Chinchilla compute-optimal 复现。不同规模训练算力不同，大模型尤其可能欠训练；同一学习率等控制条件也未针对每档调到最佳。嵌入参数在小模型中占比大、宽深同时变化、样本有限，都会影响外推。参考 [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) 和 [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，不将论文的大规模结论直接套在本实验。

## 输出、恢复与检查

输出目录默认 `artifacts/science/`，不会修改前两部分的文件：

- plan.json：语料统计、编码 token 数、模型真实参数量、种子、预算和18次实验清单。
- data/：raw、clean、val JSONL 及来源 manifest；tokens/：共享词表的二进制编码与校验。
- models/：各实验 config.json、逐步 metrics.jsonl、last.pt、best.pt 和 result.json。
- experiments.csv、report.json：汇总及拟合、预测、实测误差。
- data_quality.png、scaling_law.png、validation_curves.png：对照和训练曲线。

主指标始终是固定预算末尾的 final_val_loss，best_val_loss 只作次要记录，避免不同运行通过选择不同步数获得不公平优势。验证默认覆盖选定的完整验证语料；每250步及末尾验证，更新 best.pt。每500步或验证时保存包含优化器、scaler、抽样 RNG、Torch RNG 的 last.pt，Ctrl+C 正常终止后再次执行同一命令会从最近断点恢复，未保存部分重跑。已经完成且配置匹配的实验跳过。CPU 微型实验检查了恢复结果与连续运行一致；GPU 数值复现以服务器为准。

配置或数据改变时拒绝混用旧结果，应使用新的 science.output_dir。语料、token 缓存按 SHA256 校验。中断前尚未生成 result.json 的实验不进入最终拟合，只有所有18次完成才生成完成报告。

可选命令仅在需要分阶段查看时使用：

```bash
python main.py stat=science_prepare
python main.py stat=science_analyze
```

前者只准备语料与 token、导出计划，不训练；后者汇总已有完整结果，缺失或配置不一致会报错。science.stop_after 是恢复调试参数，正式实验保持 null。

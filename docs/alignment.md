# 第四部分：同一故事模型上的后训练对比

## 一条命令完成

不更换模型，不下载新数据，不安装新框架。使用已有 0.1B 最佳权重、完整 TinyStories 与同一份 BPE：

```bash
conda activate trackglimpse
cd ~/LLM
git pull --ff-only
python main.py stat=align
```

自动执行数据构造 → SFT → 从同一 SFT 权重分别做 DPO 和 RLVR → 统一测试四个模型 → 出图 → 汇总英文报告草稿。原始基座文件不会覆盖，后训练文件均在 `artifacts/alignment/`。默认单卡全参数微调，沿用 train.device 和 train.precision；没有多卡训练实测。

## 任务与数据

模型本来学习故事续写，本实验尝试训练它遵守一个窄约束任务，而非宣称已有通用指令能力。例如：

```text
Write a short story that uses both words: cat, tree.
Story:
```

从已有训练故事的前 144 个 token 内截取可用片段，截断尽可能落在完整句末；从片段中实际存在的两个名词构造要求，保留正文作为 chosen。将其中一个必需词的全部完整单词出现替换成非必需词，构造 rejected。重新检查正样本满足约束、负样本缺少一个必需词，不伪造人工偏好标签。

默认 4000 条训练任务、32 条开发任务、64 条测试任务。数据 SHA256 和词表来源严格校验。训练任务来自官方 train；开发和测试来自原始 val，并按无序关键词对 SHA256 mod 10 分组（0=test、1=dev、其他=train）。因此三个后训练 split 的词对不重叠，截取后的正例文本也去重。原始验证集曾用于预训练 checkpoint 选择，测试不等同于从未参与任何模型选择的全新外部数据。

偏好标签只表达“是否包含两个指定词”，不能代替人类对故事质量的判断。有限词表、片段截断和模板指令限制了任务难度及可推广性。

## 三条训练路径

| 模型 | 起点 | 默认更新 | 学习率 |
|---|---|---:|---:|
| SFT | 原有预训练 best.pt | 1000 | 3e-5 |
| SFT+DPO | SFT final.pt | 400 | 5e-6 |
| SFT+RLVR | 同一个 SFT final.pt | 120 | 5e-6 |

SFT 与 DPO 每步为 micro batch=4、grad_accum=2。各方法训练量、采样成本不同，不声称预算或 FLOPs 完全匹配。默认一个训练种子，学习率 warmup 后 cosine 衰减，weight decay=0，梯度裁剪沿用 train.grad_clip。所有阶段固定 dropout=0，保持策略采样与评分分布一致。保存和测试固定更新末尾的权重，不按测试成功率挑选 checkpoint。

### SFT

仅对 response（含 EOS）计算 next-token cross-entropy，prompt、padding 的 target 为 -100。跨 micro batch 按真实响应 token 数加权，不让 prompt loss 掩盖回答学习。

### DPO

冻结参考为 SFT。chosen/rejected 均只对响应求序列 log probability 之和，目标为：

```text
-log sigmoid(beta * ((log pi(chosen)-log pi(rejected))
                    -(log ref(chosen)-log ref(rejected))))
```

默认 beta=0.1。自动替换可能造成语法变化，回答 token 数也可能略有差异；DPO 的序列概率目标存在长度影响，不将其描述为纯粹的人类内容偏好实验。方法来源：[DPO](https://arxiv.org/abs/2305.18290)。

### RLVR

每步随机选一个训练要求，从当前 policy 采样 4 个回答，程序评分：

- 0.75 × 两个关键词的完整单词覆盖率。
- 0.15 × 词数达到 20 的指示值。
- 0.10 × 独特词比例至少 0.4 的指示值。

组内奖励减均值并除以标准差，作为 REINFORCE advantage。采样温度为 1，不做 top-k/top-p 截断；PAD/BOS 同时从采样和评分分布排除。每组只用一次 on-policy 更新，不复用旧轨迹，不使用 PPO ratio/clipping。另在生成前缀上计算当前与冻结 SFT 的完整下一 token 分布 categorical KL，默认系数 0.02。该 KL 是访问到的前缀上的正则项，不是精确计算所有故事序列的 KL。

实现为 **group-relative on-policy REINFORCE + KL**，并不冒充完整 GRPO/PPO 复现。可验证奖励是任务设置，具体策略优化器可不同。所有回答同奖励时 advantage 为零，记录 zero_advantage_groups，不人为制造不同奖励或声称一定学会。

## 统一评估与输出

基座、SFT、SFT+DPO、SFT+RLVR 在同一 64 个测试要求上，以同一 task seed 采样一次，报告成功率、词语覆盖、平均词数、重复率代理和完整生成样例。成功要求词语全覆盖、至少20词、独特词比例>=0.4。奖励只看生成正文，不把 prompt 中已出现的词算作成功。

重复率代理为 1-unique_words/words，本身与文本长度相关，也不是完整的重复 n-gram 或故事质量评价。模型可能通过列词、堆词获得奖励，需要人工阅读失败/奖励投机样例。方法效果可能相同或退化，不预设 RLVR 最佳。测试样本有限、只用一个训练种子，不作总体显著性保证。

输出：

- data/{train,dev,test}.jsonl 与 manifest.json：自动构造的数据、来源及分组规则。
- sft、dpo、rlvr：last.pt（优化器、scaler、抽样 RNG 断点）、final.pt、result.json、metrics.jsonl。
- evaluation/{base,sft,dpo,rlvr}.json：真实测试输出与验证结果。
- comparison.csv、comparison.png、report.json：方法对比和范围声明。
- `artifacts/submission/report.md`：根据第一至第四部分真实文件自动汇总的英文报告草稿，含结果图副本。尚需套用课程要求的 NeurIPS 排版，不能把 Markdown 草稿当作最终排版完成。

每100步和阶段结束保存断点，Ctrl+C 后相同命令自动从最近保存位置继续，未保存步骤重跑；已经完成且签名匹配的阶段跳过。DPO 和 RLVR 的父权重内容哈希写入签名，配置变更拒绝混用旧实验。CPU 微型测试验证了 SFT 及 RLVR 恢复与连续运行一致，GPU 复现以服务器实测为准。

正式保持 alignment.stop_after=null。可用 stat=align_prepare 只准备数据，stat=align_eval 只重新汇总/评估已有三份微调权重，stat=report 只刷新英文报告草稿；通常不必逐条执行。

## 运行后的演示

```bash
python main.py stat=web
```

页面使用原有权重和实测文件，不会从页面启动训练。参见 [页面使用说明](demo.md)。第五部分 Agent 本次裁掉，报告明确说明；本模块仍属于第四部分。

# 消融实验与工程对照结果

本文件汇总可复现的真实结果。机器可读记录位于 `evals/experiment_registry.json`。正式简历指标只能来自状态为 `completed`、样本规模充足且评测集冻结的实验。

## 当前结论

目前已完成两组工程对照和一个BM25种子基线；尚未完成正式检索消融。Seed数据只有3题，仅用于验证评测链路。

## 工程对照：页面噪声过滤

| 论文 | 元素 Before | 元素 After | 未映射 Before | 未映射 After |
|---|---:|---:|---:|---:|
| CLIP | 375 | 281 | 126 | 0 |
| BLIP-2 | 93 | 84 | 12 | 0 |

处理包括过滤重复header/page number、补齐aside/page footnote/chart类型映射，并将footnote标记为不参与检索。该结果证明数据更干净，但不能单独证明检索准确率提升。

## 工程对照：超长块拆分

| 论文 | 最大长度 Before | 最大长度 After | 超长块 Before | 超长块 After |
|---|---:|---:|---:|---:|
| CLIP | 23,283 | 2,766 | 4 | 0 |
| Full-Atom Peptide Design | 3,337 | 2,777 | 3 | 0 |
| BLIP-2 | 3,519 | 2,728 | 1 | 0 |

拆分后重复Chunk ID、悬空Parent引用和公式缺失上下文均为0。当前按字符控制长度；接入BGE-M3后还需记录真实token分布。

## BM25 Seed基线

| Cases | Recall@1 | Recall@5 | MRR |
|---:|---:|---:|---:|
| 3 | 0.3333 | 1.0000 | 0.8333 |

困难样例中，泛化查询 `theta vector field` 将附录公式排在正文公式(1)之前，体现词频检索缺少语义和篇章重要性判断。

## BGE-M3 Dense Seed基线

| Cases | Recall@1 | Recall@5 | MRR |
|---:|---:|---:|---:|
| 3 | 0.1667 | 0.6667 | 0.5000 |

中文Q-Former查询Top-1正确，说明跨语言语义召回可用；但公式相关查询弱于BM25。该结果只验证评测链路，不作为最终简历指标。

## Hybrid RRF Seed基线

| Cases | Recall@1 | Recall@5 | MRR |
|---:|---:|---:|---:|
| 3 | 0.3333 | 1.0000 | 0.7333 |

RRF将Dense的Recall@5从0.6667恢复到1.0000，并保持Q-Former跨语言查询Top-1命中；但公式目标从BM25的第2名降到第5名，因此MRR低于BM25。当前只测试了等权RRF与一组参数，不能据此断言Hybrid总体弱于BM25。

## 正式检索消融（Pending）

冻结至少15题开发集、最终扩展到30题后，统一比较：

| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | - | - | - | - | - | - |
| BGE-M3 Dense | - | - | - | - | - | - |
| BM25 + Dense + RRF | - | - | - | - | - | - |
| RRF + Reranker | - | - | - | - | - | - |

执行约束：先冻结人工标签，再运行各变体；不得看到结果后修改relevant IDs迎合模型。

## 冻结开发集检索消融（16题）

Golden Set SHA256：`6a723400e056c919d3d7fa23da85b614b9f5fe46d00290ee6cc565299cbb1bb4`

| Variant | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.1875 | 0.8047 | 0.4917 | 0.5637 |
| BGE-M3 Dense | 0.3828 | 0.7734 | 0.6406 | 0.6552 |
| BM25 + BGE-M3 + RRF | 0.4766 | 0.8438 | 0.7208 | 0.7358 |
| RRF + BGE Reranker v2-m3 | **0.5938** | **0.8984** | **0.7969** | **0.8019** |

RRF在四项指标上均为最佳；相对BM25，Recall@1提高0.2891，Recall@5提高0.0391，MRR提高0.2291，nDCG@5提高0.1721。相对Dense，nDCG@5提高0.0806。该数据集用于开发阶段消融，后续仍需独立留出集。

失败样例方面，`clip_architectures` 被BM25召回但在融合后跌出Top-5，适合用于加权RRF或Reranker优化；`peptide_evaluation_metrics` 在三种方案Top-5均未命中，但人工标注时目标表格位于BM25第10名，说明候选生成成功而最终排序不足，同样适合检验Top-20重排。

Cross-Encoder精排相对RRF进一步提升Recall@1 0.1172、Recall@5 0.0546、MRR 0.0761和nDCG@5 0.0661。`clip_architectures`恢复为Top-5命中，验证了精排价值；`peptide_evaluation_metrics`仍未命中，需要继续检查该表格是否进入RRF Top-20以及表格文本表示质量。

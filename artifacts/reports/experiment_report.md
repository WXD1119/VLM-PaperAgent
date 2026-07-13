# VLM PaperAgent Experiment Report

Generated at: 2026-07-13 12:37 UTC

## Executive Summary

- Evidence-grounded paper QA pipeline with citation validation and semantic citation judging.
- Retrieval stack compares BM25, dense retrieval, hybrid RRF and cross-encoder reranking.
- Agent memory is separated from the Paper KG: user/session state stays in memory; paper content stays in graph.
- Quality gates make hallucination-risk evaluation reproducible instead of anecdotal.

## Retrieval Ablation

Dataset: `retrieval_golden.v2.json`; cases: 48.

| Variant | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6250 | 1.0000 | 0.1679 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3_dense | 0.7083 | 1.0000 | 0.2030 | 0.6005 | 0.8420 | 0.6377 |
| BM25_BGE-M3_RRF | 0.7917 | 1.0000 | 0.2252 | 0.6606 | 0.8889 | 0.6906 |
| RRF_BGE_reranker_v2_m3 | 0.7500 | 1.0000 | 0.2099 | 0.6944 | 0.8611 | 0.6943 |

Conclusion: 在 10 篇 VLM 论文、48 个手工标注的多证据检索问题上，四种检索方案的 Hit@5 都达到 1.0，说明前 5 条结果总能召回至少一个可用证据。BM25+BGE-M3 的 RRF 融合取得最高 Hit@1=0.7917 和 MRR=0.8889，最适合作为回答前的默认证据召回方案；加入 BGE reranker 后取得最高 Recall@5=0.6944 和 nDCG@5=0.6943，说明它更擅长提升前 5 条中的多证据覆盖和排序质量。Recall@1 偏低并不代表检索不可用，而是因为每个问题通常标注多个相关 chunk，单条 top-1 很难覆盖完整证据集合。

## Answer Quality and Hallucination Risk

No answer quality JSON was provided.

## Context Guard

No context guard evaluation JSON was provided.

## Engineering Validation

| Experiment | Category | Conclusion |
|---|---|---|
| `ingestion_noise_filter_v1` | engineering_comparison | Filtering repeated page furniture removed retrieval noise and eliminated unmapped parser types. |
| `chunk_oversize_split_v1` | engineering_comparison | All chunks fit the 2800-character budget without breaking parent links or equation context. |
| `bm25_seed_v0` | retrieval_baseline | BM25 provides a working lexical baseline but ranks an appendix formula above the intended main-text equation for a generic formula query. |
| `dense_seed_v0` | retrieval_baseline | BGE-M3 retrieved the Chinese Q-Former query at rank 1, confirming cross-lingual semantic retrieval, but underperformed BM25 on the equation-focused seed cases. |
| `hybrid_rrf_seed_v0` | retrieval_baseline | RRF restored Recall@5 to 1.0 compared with dense retrieval, while preserving the cross-lingual Q-Former rank-1 hit. It did not beat BM25 MRR because the equation target moved from rank 2 to rank 5. |
| `citation_judge_smoke_v1_3` | answer_citation_evaluation | The answer-generation and independent semantic citation judge pipeline completed successfully on two supported-answer cases and one abstention case. |

## Resume-Ready Summary

- Built an evidence-grounded paper reading agent with MinerU parsing, hybrid retrieval, reranking, citation-valid answer generation, independent semantic judging, graph workspaces and memory-aware context control. Best reranked retrieval reached Hit@1=0.7500, Recall@1=0.2099, Recall@5=0.6944, nDCG@5=0.6943.

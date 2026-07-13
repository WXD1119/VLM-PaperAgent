# VLM PaperAgent Experiment Report

This report summarizes the reproducible development results for the paper-reading
agent project. Machine-readable experiment records live in
`evals/experiment_registry.json`; generated reports can be rebuilt with
`scripts/build_experiment_report.py`.

## Retrieval Ablation

Dataset: `retrieval_golden.v1.json`; 16 development queries over three papers.

| Variant | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.1875 | 0.8047 | 0.4917 | 0.5637 |
| BGE-M3 dense | 0.3828 | 0.7734 | 0.6406 | 0.6552 |
| BM25 + BGE-M3 + RRF | 0.4766 | 0.8438 | 0.7208 | 0.7358 |
| RRF + BGE reranker v2-m3 | **0.5938** | **0.8984** | **0.7969** | **0.8019** |

Conclusion: cross-encoder reranking achieved the best development-set results on
all four retrieval metrics. Compared with plain RRF, it improved Recall@1 by
0.1172, Recall@5 by 0.0546, MRR by 0.0761 and nDCG@5 by 0.0661.

## Answer Quality and Hallucination Risk

Latest validated run:

| Metric | Value |
|---|---:|
| Answers | 12 |
| Claims | 37 |
| Citation pass rate | 100.0% |
| Abstention rate | 16.7% |
| Semantic coverage rate | 100.0% |
| Semantic support rate | 100.0% |
| Unsupported claim rate | 0.0% |
| Fully supported answer rate | 83.3% |

Quality gate: **PASS**.

Interpretation:

- All 37 generated claims were checked by the independent GLM semantic citation judge.
- No unsupported or partially supported claim was found in this evaluated answer set.
- The two non-fully-supported answers are abstentions with zero factual claims, which is
  expected for insufficient-evidence questions.

## Memory and Context Control

The memory design separates agent memory from the Paper KG:

- Session memory stores short-lived task state such as current paper/workspace.
- Episodic memory stores project events.
- Profile memory stores explicit user preferences.
- Summary memory compresses older episodic turns and supports lightweight recall.
- Paper content remains in the Paper KG rather than memory.

The context guard uses session/profile memory to constrain ambiguous follow-up questions
or ask for clarification, reducing multi-turn topic drift.

## Graph Workspace

The graph layer supports a base paper graph plus personal workspace branches. Users can
add papers or curated records to a workspace without mutating the shared base graph,
similar to a lightweight Git branch model for knowledge graphs.

## Resume-Ready Summary

- Built an evidence-grounded paper reading agent with MinerU parsing, hybrid
  BM25/BGE-M3 retrieval, BGE reranking, citation-valid answer generation, independent
  GLM semantic judging, memory-aware context control and branchable paper knowledge
  graphs.
- On a 16-query development retrieval set, reranked hybrid retrieval reached
  Recall@1 0.5938, Recall@5 0.8984, MRR 0.7969 and nDCG@5 0.8019.
- On 12 generated answers with 37 claims, semantic judge coverage reached 100% with
  0% unsupported-claim rate, and the configured answer-quality gate passed.

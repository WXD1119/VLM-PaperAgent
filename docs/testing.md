# 测试与验收

## 检索指标

- Recall@K：Top-K中命中的相关证据数除以该问题的全部相关证据数。
- MRR：第一个相关证据排名倒数的宏平均。
- nDCG@5：采用二元相关性；相关证据的gain为1，按 `1/log2(rank+1)` 对靠后结果折损，再除以同一相关证据数量下的理想DCG。它同时衡量相关证据是否命中以及排序是否靠前，尤其适合一个问题对应多个证据块的场景。

## 本地/CI 快速测试

```bash
python -m pytest -v
python -m pytest --cov=paper_agent --cov-report=term-missing
```

## 真实论文流水线

```bash
python scripts/parse_paper.py \
  --pdf data/raw/BLIP2.pdf \
  --mineru-output artifacts/mineru_pipeline \
  --output artifacts/papers

python scripts/chunk_paper.py \
  --paper artifacts/papers/{paper_id}/paper.json
```

## Parser 验收项

- 页数与原 PDF 一致。
- `page` 从 1 开始，bbox 合法。
- 未映射 MinerU 类型为 0。
- header/page_number 不进入正文。
- 表格包含 Caption 与表体。
- 公式、图片、表格保留页码和章节路径。

## Chunker 验收项

- 所有 ChildChunk 能找到 ParentChunk。
- 所有 `element_ids` 能反查原始 PaperElement。
- 公式块有相邻解释上下文。
- 页脚不进入检索块。
- 相同输入重复运行生成相同 ID。
- 输出超长块数量可统计，不静默截断。

## 最近一次服务器测试

- 日期：2026-07-07
- Python：3.11.15
- 测试：9 passed / 9 collected
- 总覆盖率：84%
- Chunker：91%
- MinerU Adapter：84%

## 最近一次真实切片验收

- 三篇论文共生成 93 个 ParentChunk、319 个 RetrievalChunk。
- Parent 引用错误：0。
- 公式缺少上下文：0/46。
- 超长块：8；其中最大 23,283 字符，必须在建立向量索引前处理。

修复后重新验收：超长块、重复 Chunk ID、无效 Parent 引用和公式缺失上下文均为0。

## BM25 Seed 基线

- Cases：3（仅验证工具链，不作为简历性能指标）。
- Recall@1：0.3333。
- Recall@5：1.0000。
- MRR：0.8333。
- 输出：`artifacts/evals/bm25.seed.json`。

## Graph workspace validation

Graph workspaces are validated through invariants rather than screenshots:

- forking a workspace must not rewrite the base workspace files;
- the effective graph must contain all visible base nodes plus the user's delta nodes;
- tombstoned nodes and tombstoned edges must be hidden from the effective graph;
- edges whose endpoints are hidden must also be hidden;
- node IDs and edge IDs must remain unique after all deltas are applied;
- every edge endpoint must exist in the effective graph;
- every `Paper` node must still have at least one `Chunk`;
- every non-abstained `Claim` must have at least one `SUPPORTED_BY` edge;
- every `SUPPORTED_BY` edge must point to a `Chunk`;
- `diff` between base and fork must report only the fork's added/removed graph records.

Planned smoke tests:

```bash
python scripts/fork_graph.py \
  --base artifacts/graph \
  --output artifacts/graph_workspaces/ws_demo

python scripts/inspect_workspace.py \
  --workspace artifacts/graph_workspaces/ws_demo \
  --show-errors

python scripts/diff_graph.py \
  --base artifacts/graph \
  --workspace artifacts/graph_workspaces/ws_demo
```

These commands are design targets for the next implementation phase; the current stable
graph validation entry point remains:

```bash
python scripts/inspect_graph.py --graph artifacts/graph --show-errors
```

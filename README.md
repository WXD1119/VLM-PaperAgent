# VLM-PaperAgent

面向 CV/VLM 论文的证据可追溯精读 Agent。核心目标不是堆叠 Agent 数量，而是让每个回答、主张与审稿疑点都能定位到论文页码、章节、公式或图表。

## MVP 闭环

`PDF -> 结构化解析 -> 混合检索 -> Claim/Evidence 审稿 -> 带引用报告`

当前仓库提供领域模型、基础设施端口、手写 FSM 调度器、API/UI 入口骨架和测试。Marker/MinerU、Chroma、Redis、Neo4j 等实现按接口逐步接入。

## 快速验证

```powershell
python -m pytest
```

## 转换 MinerU 解析结果

MinerU 建议独立部署；业务环境只读取其 `*_content_list.json`：

```bash
python scripts/parse_paper.py \
  --pdf data/raw/BLIP2.pdf \
  --mineru-output artifacts/mineru_pipeline \
  --output artifacts/papers
```

输出位于 `artifacts/papers/{paper_id}/paper.json`，其中 MinerU 的 0-based
`page_idx` 已转换为 1-based `page`，并补充章节路径、父级与相邻元素链接。

生成公式感知的父子切片：

```bash
python scripts/chunk_paper.py \
  --paper artifacts/papers/{paper_id}/paper.json
```

切片器将公式、表格和图片保留为独立证据块；公式块绑定同章节前后解释，
普通文本按字符预算合并，并保留页码、章节和原始元素 ID。

建立内存 BM25 基线并检索全部论文：

```bash
python scripts/search_bm25.py \
  --chunks artifacts/papers \
  --query "flow matching objective" \
  --top-k 5
```

可使用 `--paper-id` 限定论文，或使用 `--kind equation` 仅检索公式。结果始终
返回 `paper_id`、页码、章节和 `chunk_id`，可直接作为后续答案引用证据。

运行首版检索 Golden Set：

```bash
python scripts/evaluate_bm25.py \
  --chunks artifacts/papers \
  --golden evals/retrieval_golden.seed.json \
  --output artifacts/evals/bm25.seed.json
```

`seed` 文件只有3个经人工核对的开发样例，用于验证评测链路，不能作为简历指标。
正式对比前需扩展到至少30个问题，并冻结数据集版本。

交互式人工标注首批15题：

```bash
python scripts/annotate_retrieval.py \
  --chunks artifacts/papers \
  --questions evals/retrieval_questions.v1.json \
  --seed evals/retrieval_golden.seed.json \
  --output evals/retrieval_golden.v1.json \
  --annotator wxd
```

工具逐题展示BM25 Top-10，输入 `1,3-4` 选择相关候选，`s`跳过，`q`安全退出。
每确认一题立即原子写盘，可以随时中断后继续。BM25只负责候选生成，最终相关性必须由人工确认。

## Dense Retrieval + Chroma

默认模型为 `BAAI/bge-m3`。业务代码显式计算归一化Embedding并传给Chroma，避免使用
Chroma默认模型。首次建库：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/index_dense.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0
```

语义检索：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/search_dense.py \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --query "How are visual features connected to the language model?"
```

用与BM25完全相同的Golden Set评测Dense：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_dense.py \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --golden evals/retrieval_golden.seed.json \
  --output artifacts/evals/dense.seed.json
```

Hybrid Top-20经过多语言Cross-Encoder精排：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_reranked.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --embedding-model artifacts/models/bge-m3 \
  --reranker-model artifacts/models/bge-reranker-v2-m3 \
  --offline \
  --device cuda:0 \
  --golden evals/retrieval_golden.v1.json \
  --candidate-k 20 \
  --top-k 5 \
  --output artifacts/evals/reranked.v1.json
```

## 目录

- `src/paper_agent/domain`：稳定的业务数据契约。
- `src/paper_agent/workflow`：状态、合法转换、重试和调度。
- `src/paper_agent/ingestion`：PDF 解析、规范化和父子切片。
- `src/paper_agent/retrieval`：Dense、BM25、RRF 和 Rerank。
- `src/paper_agent/agents`：Claim、Evidence、Critic、Judge、Reporter 节点。
- `src/paper_agent/storage`：Chroma、Redis、Neo4j 适配器端口。
- `src/paper_agent/evaluation`：检索、引用和审稿评测。
- `src/paper_agent/api`：FastAPI 服务入口。
- `app`：Streamlit 演示入口。

## 工程原则

1. 所有结论必须携带 `evidence_ids`。
2. Agent 只返回结构化增量，不直接控制任意状态跳转。
3. 状态转换集中校验，节点执行支持重试和 checkpoint。
4. 外部存储通过 Protocol 隔离，单元测试不依赖 Docker。
5. 指标只使用真实 Golden Set 评测结果，不预填宣传数字。

实验结果统一登记在 `evals/experiment_registry.json`，可读汇总位于
`docs/ablation-results.md`；Seed指标仅验证工具链，不作为简历KPI。

## 环境

source /workspace/guest/wxd/anaconda3/etc/profile.d/conda.sh
conda activate paper-agent
cd /workspace/guest/wxd/VLM-PaperAgent
## Evidence-grounded answer generation

The answer stage uses reranked chunks as a bounded `EvidencePack`. The LLM must return
claim-level evidence IDs; the citation validator rejects IDs that were not supplied. The default
provider is configurable through an OpenAI-compatible interface, so DeepSeek and GLM can be
compared without changing agent logic.

Answer generation is transient by default. The CLI writes immutable answer artifacts for
review and evaluation, but those artifacts do not enter the paper knowledge graph by
default.

```bash
pip install -e '.[retrieval,local-vlm]'
CUDA_VISIBLE_DEVICES=2,3 python scripts/answer_question.py \
  --chunks artifacts/papers --offline --device cuda:0 --llm-device cuda:1 \
  --provider local --model artifacts/models/Qwen3-VL-8B-Instruct \
  --query 'How does Q-Former connect the frozen image encoder and LLM?'
```

API keys are read from environment variables and must never be committed. The first MVP sends
MinerU text, equations, tables, and figure captions to a text LLM. Raw figure pixels will be routed
to a vision model only when caption/OCR evidence is insufficient.

## One-command demo

`scripts/ask.py` is the presentation-friendly entry point. It runs hybrid retrieval, optional
cross-encoder reranking, local answer generation, citation integrity validation, and writes the
immutable answer bundle for later human or GLM judging. It does not automatically write the
answer into a graph workspace.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
python scripts/ask.py \
  --query "How does Q-Former bridge the frozen image encoder and frozen language model?" \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --embedding-model artifacts/models/bge-m3 \
  --reranker-model artifacts/models/bge-reranker-v2-m3 \
  --model artifacts/models/Qwen3-VL-8B-Instruct \
  --offline \
  --device cuda:0 \
  --llm-device cuda:1 \
  --output artifacts/answers/demo.qformer.answer.json
```

Use `--no-rerank` for a faster RRF-only smoke test, `--paper-id` to restrict a query to one paper,
and `--kind equation|table|figure|text` to inspect modality-specific evidence.

When the local GLM judge service is running, add `--judge-url` to turn the demo into a guarded
generate-then-verify loop. Unsupported claims are fed back to the answer model for revision; if the
answer still cannot pass after `--max-answer-attempts`, the script emits a safe abstention instead
of publishing an unsupported answer.

```bash
python scripts/ask.py \
  --query "How does Q-Former bridge the frozen image encoder and frozen language model?" \
  --offline \
  --device cuda:0 \
  --llm-device cuda:1 \
  --judge-url http://127.0.0.1:8765 \
  --max-answer-attempts 2
```

## FastAPI service

The same pipeline can be exposed as a local HTTP service. Runtime configuration is read from
environment variables so model paths do not need to be hard-coded into requests.

```bash
export PAPER_AGENT_CHUNKS=artifacts/papers
export PAPER_AGENT_CHROMA_DB=artifacts/chroma
export PAPER_AGENT_EMBEDDING_MODEL=artifacts/models/bge-m3
export PAPER_AGENT_RERANKER_MODEL=artifacts/models/bge-reranker-v2-m3
export PAPER_AGENT_LLM_MODEL=artifacts/models/Qwen3-VL-8B-Instruct
export PAPER_AGENT_OFFLINE=1
export PAPER_AGENT_DEVICE=cuda:0
export PAPER_AGENT_LLM_DEVICE=cuda:1
export PAPER_AGENT_JUDGE_URL=http://127.0.0.1:8765

uvicorn paper_agent.api.main:app --host 127.0.0.1 --port 8000
```

Then ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does Q-Former bridge the frozen image encoder and frozen language model?", "top_k": 5}'
```

## Lightweight evidence graph

The first graph backend is a reproducible JSONL intermediate representation rather than a required
Neo4j service. Its default mode is a paper knowledge graph: it records paper, section, chunk and
concept nodes plus paper-structure and mention edges such as `HAS_SECTION`, `HAS_CHUNK`,
`CONTAINS` and `MENTIONS`.

Build a graph from all parsed papers and chunks:

```bash
python scripts/build_graph.py \
  --papers artifacts/papers \
  --output artifacts/graph
```

Legacy traceability demos can still include saved answer bundles with `--include-answers`, but new
workflows should keep generated answers in artifact memory rather than mixing user/agent QA records
into the paper graph.

Inspect and validate graph invariants:

```bash
python scripts/inspect_graph.py --graph artifacts/graph --show-errors
```

Query the graph:

```bash
python scripts/query_graph.py --graph artifacts/graph --papers

python scripts/query_graph.py \
  --graph artifacts/graph \
  --paper-id paper_6bc5d399d6127a64 \
  --chunks \
  --limit 5

python scripts/query_graph.py --graph artifacts/graph --claim-supports --limit 10

python scripts/query_graph.py --graph artifacts/graph --search "Q-Former" --node-type Claim

python scripts/query_graph.py --graph artifacts/graph --concept "Q-Former"
```

Correctness is checked by graph invariants rather than visual inspection: node IDs and edge IDs
must be unique, every edge endpoint must exist, every paper must link to chunks, and `MENTIONS`
edges must connect supported paper-content sources to `Concept` targets. Legacy answer-in-graph
mode additionally validates claim-to-evidence edges. The JSONL files can later be imported into
Neo4j/Cypher without changing the graph-building logic.

Concept lookup keeps disambiguation explicit. Exact concept IDs are expanded directly; ambiguous
keywords return candidate concept nodes instead of silently choosing the wrong sense.

## Graph workspaces and branching

The next graph milestone is multi-user workspace branching. A user should be able to
start an empty paper graph or fork an existing graph, then add papers without mutating
the original graph. The planned design uses immutable
graph commits plus copy-on-write JSONL deltas, similar to a lightweight Git model for
evidence graphs.

See `docs/graph-workspaces.md` for the implementation plan, validation invariants,
API sketch and the optional Neo4j upgrade path.

Create a local workspace fork:

```bash
python scripts/fork_graph.py \
  --base artifacts/graph \
  --workspace-id ws_wxd_demo \
  --owner wxd \
  --output artifacts/graph_workspaces/ws_wxd_demo
```

Inspect its effective graph:

```bash
python scripts/inspect_workspace.py \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --show-errors
```

The P1 workspace implementation stores a base graph pointer plus immutable workspace commits.
The effective graph is loaded as `base graph + workspace deltas - tombstones`; base JSONL files
are not rewritten by fork operations.

Commit a graph delta into a workspace:

```bash
python scripts/commit_graph_delta.py \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --nodes artifacts/tmp/new_nodes.jsonl \
  --edges artifacts/tmp/new_edges.jsonl \
  --author wxd \
  --message "add private graph records"
```

Compare the base graph with the workspace effective graph:

```bash
python scripts/diff_graph.py \
  --base artifacts/graph \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --show-ids
```

Delta commits are validated before they are written. Use `--allow-invalid` only for debugging
conflict or tombstone scenarios that intentionally break the effective graph invariants.

Add parsed papers directly to a workspace:

```bash
python scripts/add_paper_to_workspace.py \
  --workspace artifacts/graph_workspaces/ws_wxd_demo \
  --paper artifacts/papers/{paper_id}/paper.json \
  --chunks artifacts/papers/{paper_id}/chunks.json \
  --author wxd

```

These commands build graph fragments from normal project artifacts, diff them against the
workspace effective graph, and commit only new records as workspace deltas. Ordinary
questions and generated answers stay in agent memory or immutable answer artifacts.

## Agent memory

Agent memory is separate from the paper graph. The current lightweight memory layer has:

- short-term session memory: current task, paper, workspace and last answer path;
- episodic memory: append-only event history;
- user profile memory: stable preferences and environment defaults;
- artifact memory: immutable answer/evaluation files.

Useful commands:

```bash
python scripts/list_answers.py --answers artifacts/answers

python scripts/memory_profile.py --set default_workspace_id ws_wxd_demo

python scripts/memory_log_event.py \
  --event-type paper_added \
  --summary "Added LLaVA.pdf to wxd workspace" \
  --payload-json '{"paper_id":"paper_dc8bace378a282ea"}'
```

See `docs/agent-memory.md` for the memory design and its boundary with the paper graph.

## Product service plan

The product-facing design separates lower-level modules from user-facing services.
In particular, `Evidence Graph Store` is the graph foundation, while `Graph-grounded QA`
and `Concept Graph Explorer` are two different user features built on top of that graph:

- `Graph-grounded QA`: question in, paragraph answer with citations out.
- `Concept Graph Explorer`: keyword in, structured concept neighborhood out.

See `docs/product-service-plan.md` for the current service map and implementation order.

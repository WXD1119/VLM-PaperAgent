# VLM-PaperAgent

VLM-PaperAgent 是一个面向视觉语言模型论文的证据可追溯阅读 Agent。它的核心目标不是堆叠“多 Agent”“企业级架构”这类口号，而是让每个回答、每条 claim、每个实验结论都能追溯到论文中的页码、章节、公式、表格或图注。

项目当前已经形成一条完整闭环：

```text
PDF 解析 -> 结构化 Paper/Chunk -> BM25 + Dense + RRF + Rerank 检索
-> Evidence Pack -> LLM 回答 -> 引用完整性校验 -> 语义 Judge -> Memory/Graph 记录
```

## 当前亮点

- 证据可追溯回答：答案按 claim 绑定 evidence ID，并校验 evidence ID 是否真实来自检索结果。
- 混合检索与精排：支持 BM25、BGE-M3 dense retrieval、RRF 融合、BGE reranker。
- 多证据评测：自建 10 篇 VLM 论文、48 个问题的人工标注 retrieval golden set。
- 幻觉风险控制：使用独立 judge LLM 对 claim-evidence 支持关系做语义审查。
- 轻量 Paper KG：保存论文、章节、chunk、concept 关系；用户记忆和生成答案默认不写入论文图谱。
- 三层 memory 骨架：session / episodic / profile 分离，并用 context guard 处理模糊追问。
- 工程可复现：实验 registry、自动报告、pytest、Chroma 索引一致性检查。

## 最新检索实验结果

数据集：`evals/retrieval_golden.v2.json`，覆盖 10 篇 VLM 论文、48 个手工标注问题。

| 方法 | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6250 | 1.0000 | 0.1679 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3 dense | 0.7083 | 1.0000 | 0.2030 | 0.6005 | 0.8420 | 0.6377 |
| BM25 + BGE-M3 RRF | 0.7917 | 1.0000 | 0.2252 | 0.6606 | 0.8889 | 0.6906 |
| RRF + BGE reranker | 0.7500 | 1.0000 | 0.2099 | 0.6944 | 0.8611 | 0.6943 |

中文结论：

- 所有方案 `Hit@5=1.0`，说明前 5 条结果总能找到至少一个可用证据。
- `BM25 + BGE-M3 RRF` 的 `Hit@1` 和 `MRR` 最好，适合作为默认在线召回方案。
- `RRF + BGE reranker` 的 `Recall@5` 和 `nDCG@5` 最好，适合高质量、多证据回答模式。
- `Recall@1` 偏低不是系统不可用，而是因为一个问题经常对应多个证据 chunk，top-1 很难覆盖完整证据集合。

## 快速开始

推荐在 Linux/远程服务器运行，Python 版本建议 3.11。

```bash
conda create -n paper-agent python=3.11 -y
conda activate paper-agent
pip install -e '.[retrieval,llm,api,local-vlm,dev]'
```

如果只跑单元测试：

```bash
python -m pytest -v
```

## 1. 解析论文

MinerU 建议独立环境部署，业务环境只读取 MinerU 产出的 `*_content_list.json`。

```bash
python scripts/parse_paper.py \
  --pdf data/raw/BLIP2.pdf \
  --mineru-output artifacts/mineru_pipeline \
  --output artifacts/papers
```

生成 chunk：

```bash
python scripts/chunk_paper.py \
  --paper artifacts/papers/{paper_id}/paper.json
```

输出结构：

```text
artifacts/papers/{paper_id}/paper.json
artifacts/papers/{paper_id}/chunks.json
```

chunk 会保留：

- `paper_id`
- `chunk_id`
- 页码
- 章节路径
- 原始 element IDs
- 文本 / 公式 / 表格 / 图注类型

## 2. 构建 Dense 索引

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/index_dense.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --reset
```

预期输出：

```text
indexed: 863
collection_count: 863
```

`--reset` 会删除并重建 Chroma collection，避免旧索引残留。

## 3. 检索评测

BM25：

```bash
python scripts/evaluate_bm25.py \
  --chunks artifacts/papers \
  --golden evals/retrieval_golden.v2.json \
  --output artifacts/evals/bm25.v2.json
```

Dense：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_dense.py \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --golden evals/retrieval_golden.v2.json \
  --output artifacts/evals/dense.v2.json
```

Hybrid：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_hybrid.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0 \
  --golden evals/retrieval_golden.v2.json \
  --candidate-k 20 \
  --rrf-k 60 \
  --top-k 5 \
  --output artifacts/evals/hybrid.v2.json
```

Reranked：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/evaluate_reranked.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --embedding-model artifacts/models/bge-m3 \
  --reranker-model artifacts/models/bge-reranker-v2-m3 \
  --offline \
  --device cuda:0 \
  --golden evals/retrieval_golden.v2.json \
  --candidate-k 20 \
  --top-k 5 \
  --output artifacts/evals/reranker.v2.json
```

## 4. 端到端 Demo

`scripts/ask.py` 是面试展示时最推荐的入口。它会执行：

```text
context guard -> hybrid retrieval -> rerank -> answer generation
-> citation validation -> optional semantic judge -> memory summary
```

基础版：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
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

如果 GLM judge 服务已经启动，可以打开语义引用 gate：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
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
  --judge-url http://127.0.0.1:8765 \
  --max-answer-attempts 2 \
  --output artifacts/answers/demo.qformer.gated.answer.json
```

如果 judge 判定有 unsupported claim，`ask.py` 会把反馈传回 answer agent 重试；多次失败后安全 abstain，而不是输出无证据支持的答案。

## 5. API 服务

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

请求：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How does Q-Former bridge the frozen image encoder and frozen language model?", "top_k": 5}'
```

## 6. Paper Knowledge Graph

默认图谱只保存论文内容，不保存用户画像、用户偏好或普通聊天。

```bash
python scripts/build_graph.py \
  --papers artifacts/papers \
  --output artifacts/graph
```

验证：

```bash
python scripts/inspect_graph.py \
  --graph artifacts/graph \
  --show-errors
```

查询：

```bash
python scripts/query_graph.py --graph artifacts/graph --papers
python scripts/query_graph.py --graph artifacts/graph --concept "Q-Former"
python scripts/query_graph.py --graph artifacts/graph --search "Q-Former" --node-type Claim
```

## 7. Memory 设计

Agent memory 和 Paper KG 分离：

- Session memory：当前论文、上一个问题、短期任务状态。
- Episodic memory：执行过的重要事件，例如生成了某个 answer artifact。
- Profile memory：用户长期偏好，例如回答语言。
- Artifact memory：答案 JSON、评测报告等可复现文件。
- Paper KG：只保存论文结构和论文内容，不保存用户个人信息。

这种设计避免把用户行为、模型回答和论文事实混在同一个知识图谱里。

## 目录结构

```text
src/paper_agent/domain          领域模型：Paper、Chunk、Answer、Workflow
src/paper_agent/ingestion       MinerU 输出适配、论文结构化、chunk 构建
src/paper_agent/retrieval       BM25、Dense、RRF、Reranker
src/paper_agent/agents          AnswerAgent、CitationValidator、SemanticJudge
src/paper_agent/evaluation      检索、引用、上下文守卫、答案质量评测
src/paper_agent/graph           轻量 Paper KG 和 workspace 分支
src/paper_agent/memory          session / episodic / profile memory
src/paper_agent/api             FastAPI 服务
scripts                         CLI 工具和实验脚本
evals                           golden set、实验 registry、benchmark manifest
docs                            设计文档、实验记录、项目说明
```

## 重要文档

- `docs/vlm-benchmark-v2.md`：VLM benchmark v2 数据集和检索结果。
- `docs/answer-quality-evaluation.md`：答案质量与 hallucination 风险评估。
- `docs/agent-memory.md`：Agent memory 设计。
- `docs/graph-workspaces.md`：图谱 workspace / 分支机制。
- `artifacts/reports/experiment_report.md`：自动生成的阶段性实验报告。

## 简历版一句话

构建了一个面向 VLM 论文的 evidence-grounded reading agent，支持 MinerU 论文解析、BM25+BGE-M3+RRF+BGE-reranker 混合检索、claim-level citation validation、独立 LLM judge、轻量 Paper KG、memory-aware context guard，并在 10 篇论文 48 个手工标注问题上完成检索消融：Hybrid 达到 Hit@1=0.7917 / MRR=0.8889，Reranker 达到 Recall@5=0.6944 / nDCG@5=0.6943。

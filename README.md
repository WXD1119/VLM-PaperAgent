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
immutable answer bundle for later human or GLM judging.

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

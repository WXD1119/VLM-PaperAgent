# VLM-PaperAgent 代码学习路线

> 学习原则：按“数据 → 检索 → 回答 → Agent 工作流 → 产品与基础设施”阅读，而不是按目录字母顺序阅读。
>
> 不建议一开始阅读 `api/main.py`。它是组装层，先理解底层对象和主链路后再看会高效得多。

## 1. 技术栈总览

| 层级 | 技术/组件 | 项目作用 |
|---|---|---|
| 基础语言与建模 | Python 3.11、Pydantic v2 | 定义论文、Chunk、答案、计划、Trace 等结构化数据 |
| PDF 解析 | MinerU、OCR/PDF 结构化解析 | 将论文解析为标题、正文、公式、表格、图片等元素 |
| 检索 | BM25、BGE-M3、Chroma、RRF | 稀疏检索、向量检索与融合召回 |
| 精排 | BGE Reranker / Cross Encoder | 对候选 Chunk 做 query-document 精排 |
| 回答模型 | Qwen3-VL-8B-Instruct、Transformers、Accelerate | 本地生成带引用的论文答案 |
| 审查模型 | GLM-4.1V-9B-Thinking | 独立语义 Judge，判断 Claim 是否被证据支持 |
| Agent 工作流 | LangGraph | 固定执行上下文解析、路由、检索、回答、校验、重写/拒答 |
| 路由与规划 | Query Router、Evidence Planner | 按问题类型选择文本、公式、表格、图示等证据策略 |
| 记忆 | Redis、MySQL、本地 JSONL | 会话、滑动窗口、摘要、用户偏好、长期事件 |
| 图谱 | 本地 JSONL Graph、Neo4j | 论文概念、章节、Chunk、关系查询与候选发现 |
| 服务层 | FastAPI、Uvicorn、SSE、Streamlit | API、实时进度、网页问答、上传、图谱界面 |
| 工具治理 | Tool Gateway、Scope、超时与重试 | 限制工具调用权限，禁止模型任意执行 |
| 质量工程 | Pytest、Golden Set、LLM-as-Judge、Trace | 自动评测、回归测试、失败归因、可观测性 |
| 部署 | Docker Compose、环境变量 | Redis、MySQL、Neo4j 等服务编排 |

## 2. 先建立全流程地图

优先阅读：

1. `README.md`
2. `docs/system-architecture.png`
3. `docs/query-routing.md`
4. `scripts/ask.py`

先记住主链路，不要急于理解全部实现：

```text
用户问题
→ Context Guard
→ Query Router
→ Evidence Planner
→ 检索/精排
→ Evidence Pack
→ Answer Agent
→ Citation Validator
→ Semantic Judge
→ Trace / Memory
```

## 3. 第一阶段：领域数据模型

阅读顺序：

1. `src/paper_agent/domain/chunk.py`
2. `src/paper_agent/domain/paper.py`
3. `src/paper_agent/domain/answer.py`
4. `src/paper_agent/domain/review.py`

重点对象：

```text
Paper → 一篇论文
RetrievalChunk → 可检索、可引用的证据块
EvidencePack → 当前问题的候选证据集合
GroundedAnswer → 带 Claim 的回答
CitationValidation / SemanticCitationReport → 两层校验结果
```

学习目标：理解项目中各模块传递的到底是什么数据，而不是先理解模型调用。

测试：

```bash
python -m pytest tests/test_review_models.py -v
```

## 4. 第二阶段：论文解析与 Chunk

阅读顺序：

1. `src/paper_agent/ingestion/mineru_adapter.py`
2. `src/paper_agent/ingestion/normalizer.py`
3. `src/paper_agent/ingestion/chunker.py`
4. `src/paper_agent/ingestion/element_linker.py`

需要回答的问题：

```text
论文 PDF 的标题、段落、公式、表格、图片如何变为 paper.json？
为什么 Child Chunk 要保留 parent、页码、章节路径和公式上下文？
为什么表格、公式、图示不能简单和普通正文混在一起？
```

测试：

```bash
python -m pytest tests/test_mineru_adapter.py tests/test_chunker.py -v
```

## 5. 第三阶段：检索与精排

阅读顺序：

1. `src/paper_agent/retrieval/sparse.py`：BM25。
2. `src/paper_agent/retrieval/embedding.py`：BGE-M3 编码。
3. `src/paper_agent/retrieval/dense.py`：向量检索。
4. `src/paper_agent/retrieval/fusion.py`：RRF 融合。
5. `src/paper_agent/retrieval/hybrid.py`：混合检索。
6. `src/paper_agent/retrieval/reranker.py`：交叉编码器精排。
7. `src/paper_agent/storage/chroma_store.py`：Chroma 持久化封装。

核心概念：

```text
BM25：关键词精确匹配强
Dense：语义与中英混合查询强
RRF：融合两个排序列表
Reranker：对 query 与候选 Chunk 做更精确的相关性判断
```

对应评测脚本：

```bash
python scripts/evaluate_bm25.py ...
python scripts/evaluate_dense.py ...
python scripts/evaluate_hybrid.py ...
python scripts/evaluate_reranked.py ...
```

这部分要结合已有的 Hit@k、Recall@k、MRR、nDCG@k 实验结果一起学习。

## 6. 第四阶段：答案生成与可信引用

阅读顺序：

1. `src/paper_agent/llm/client.py`
2. `src/paper_agent/agents/answer.py`
3. `src/paper_agent/security/evidence.py`

重点区分两层校验：

```text
Citation Validator
= 确定性规则校验：Claim 是否引用了存在的 Evidence ID

Semantic Judge
= 语义校验：Evidence 是否真的支持该 Claim
```

“有引用”不等于“引用正确”。

测试：

```bash
python -m pytest tests/test_answer_agent.py tests/test_citation_evaluation.py -v
```

## 7. 第五阶段：LangGraph、路由与证据规划

阅读顺序：

1. `src/paper_agent/runtime/research_graph.py`
2. `src/paper_agent/routing/models.py`
3. `src/paper_agent/routing/query_classifier.py`
4. `src/paper_agent/routing/evidence_planner.py`

需要理解：项目不是开放式 ReAct，而是受限状态图。

```text
状态图决定流程
工具网关决定权限
Pydantic Schema 决定输入输出边界
Judge 决定是否重写或安全拒答
```

路由示例：

```text
公式问题 → equation + text
表格问题 → table + text
图示问题 → figure + text
机制问题 → text + figure
多跳关系 → 图谱候选发现 + 原始 Chunk 引用
定义问题 → 轻量检索，可跳过精排/Judge
```

测试：

```bash
python -m pytest \
  tests/test_langgraph_runtime.py \
  tests/test_routing.py \
  tests/test_routing_evaluation.py \
  -v
```

## 8. 第六阶段：记忆与上下文漂移控制

阅读顺序：

1. `src/paper_agent/memory/session.py`
2. `src/paper_agent/memory/context.py`
3. `src/paper_agent/memory/summary.py`
4. `src/paper_agent/memory/entity_resolution.py`
5. `src/paper_agent/memory/backends.py`

重点：

```text
短期状态：当前论文、当前实体
滑动窗口：最近对话
摘要记忆：压缩历史，并保留 evidence_chunk_ids
长期记忆：用户偏好、事件、计划
```

理解 Redis、MySQL、本地 JSON 的分工，以及为什么 Paper KG 中不能保存用户偏好和生成回答。

## 9. 第七阶段：图谱与 Neo4j

阅读顺序：

1. `src/paper_agent/graph/model.py`
2. `src/paper_agent/graph/builder.py`
3. `src/paper_agent/graph/query.py`
4. `src/paper_agent/graph/workspace.py`
5. `src/paper_agent/storage/neo4j_store.py`

关键原则：

```text
Neo4j：发现关系和候选节点
Chunk：提供论文原文、章节和页码
最终回答：引用 Chunk，而非图谱边
```

## 10. 第八阶段：工程化、工具、安全、API 与前端

阅读顺序：

1. `src/paper_agent/security/boundaries.py`
2. `src/paper_agent/tools/orchestrator.py`
3. `src/paper_agent/tools/gateway.py`
4. `src/paper_agent/observability/tracing.py`
5. `src/paper_agent/api/main.py`
6. `app/streamlit_app.py`

学习重点：

```text
为什么工具要有 Scope、超时、重试和资源冲突控制？
为什么 Trace 只记录哈希、ID、耗时和决策，而不保存论文原文？
为什么 SSE 只展示执行状态，不展示模型原始思维过程？
为什么 API 是组装层，应该最后读？
```

## 11. 一周最短学习路径

如果目标是一周内看懂项目主线，只读下面文件：

```text
1. domain/chunk.py、domain/answer.py
2. ingestion/chunker.py
3. retrieval/hybrid.py、retrieval/reranker.py
4. agents/answer.py
5. runtime/research_graph.py
6. routing/query_classifier.py、routing/evidence_planner.py
7. memory/context.py
8. api/main.py
```

每读完一个模块，立刻阅读对应 `tests/test_*.py`。测试代码通常比主代码短，最适合用来确认输入、输出、边界条件和设计意图。

## 12. 学习完成的自检问题

完成这条路线后，应该能清楚回答：

1. 一篇 PDF 如何变成可引用的 Chunk？
2. BM25、Dense、RRF、Reranker 分别解决什么问题？
3. 为什么答案需要 Citation Validator 和独立 Semantic Judge 两层校验？
4. LangGraph 的状态图如何限制 Agent 的执行边界？
5. Query Router 为什么能降低成本和延迟？
6. 记忆、图谱、论文事实为什么必须隔离存储？
7. 系统如何通过 Golden Set、Trace 和质量门禁持续迭代？

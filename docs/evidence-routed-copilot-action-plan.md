# Evidence-Grounded Research Copilot：行动方案与验收指标

> 定位：面向算法研发团队的可追溯论文调研与知识沉淀系统。
>
> 核心目标：从“能回答论文问题”的 RAG 工具，升级为“会判断问题、会选择证据策略、会诊断失败、会回测改进”的 Agent 系统。

## 1. 不变的设计原则

1. **论文事实与用户状态隔离。** Neo4j Paper KG 仅保存论文、章节、Chunk、概念和论文内关系；用户偏好、会话、计划、生成答案和运行记录保存到 Redis、MySQL、文件工件或 Trace 存储。
2. **最终引用必须回到原始论文证据。** 图谱只帮助发现实体、关系与候选 Chunk；最终答案只能引用带页码的原始 Chunk，不能引用 Neo4j 边或会话摘要。
3. **模型不拥有任意执行权。** 工具调用经过白名单、输入输出 Schema、权限、超时、重试和资源限制；不能由 LLM 直接运行 Shell、上传文件、写图谱或访问任意网络。
4. **低置信度不猜。** 指代不清、论文范围不明、证据不足或多次修复失败时，澄清或拒答优先于编造答案。
5. **线上策略变更先离线回测。** Monitor 只能产出诊断和候选配置；检索权重、阈值、Prompt 和安全策略必须经过评测与人工确认后发布。

## 2. 目标架构

```text
用户问题
  ↓
Context Guard / Entity Resolver
  ↓
Query Router（意图、范围、证据类型、执行策略）
  ├─ 单篇证据问答 ────────┐
  ├─ 多论文调研计划 ──────┤
  ├─ 图谱关系查询 ────────┤
  ├─ 图/表/公式定向查询 ──┤
  └─ 模糊追问澄清 ────────┘
  ↓
Evidence Planner
  ↓
并行检索（BM25 / Dense / 图谱候选 / 图表公式）
  ↓
Evidence Merge / Dedup / BGE Reranker
  ↓
Evidence Pack（原始 Chunk + 页码 + 章节）
  ↓
Answer Agent
  ↓
Citation Validator（确定性） → Semantic Judge（语义）
  ↓
诊断驱动修复：重写 / 再检索 / 澄清 / 拒答
  ↓
Trace + Memory + Artifact + 离线评测
```

## 3. 当前基础

项目已经具备：

- MinerU 论文解析、结构化 `paper.json`、多模态 Chunk。
- BM25、BGE-M3 Dense、RRF Hybrid、BGE Reranker 检索消融。
- 证据包、Claim 引用完整性校验、独立语义 Judge 与安全拒答。
- Context Guard、三层记忆骨架、会话摘要压缩、Redis/MySQL/Neo4j 存储边界。
- LangGraph 固定证据问答流程、受控工具网关、超时与重试策略。
- 图谱概念查询、图谱 Workspace、受限多论文 Plan-and-Execute。
- FastAPI、Streamlit、SSE 进度展示、Trace、上传解析任务与 Worker。
- 10 篇 VLM 论文、48 条人工标注检索问题，以及检索/回答质量报告。

已知 v2 检索结果（48 条人工标注问题）：

| 方案 | Hit@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.6250 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3 Dense | 0.7083 | 0.6005 | 0.8420 | 0.6377 |
| Hybrid RRF | 0.7917 | 0.6606 | 0.8889 | 0.6906 |
| RRF + Reranker | 0.7500 | 0.6944 | 0.8611 | 0.6943 |

回答质量当前基线：12 个答案、37 个 Claim；引用完整性通过率 100%、语义覆盖率 100%、语义支持率 100%、无支持 Claim 比例 0%、完全支持答案比例 83.3%。

## 4. P0：问题路由与证据类型路由

### 4.1 目标

不再让所有问题走同一条“Hybrid → Rerank → Answer → Judge”固定链路，而是根据问题类型选择最小充分的证据策略，以降低无效检索、GPU 成本和延迟。

### 4.2 新增模块

```text
src/paper_agent/routing/
  models.py              # QueryPlan、意图、证据类型、置信度
  query_classifier.py    # 规则优先、LLM 可选的分类器
  query_rewriter.py      # 仅在需要时改写查询
  retrieval_policy.py    # 将 QueryPlan 映射为受控检索策略
  evaluator.py           # 路由离线评测
```

### 4.3 QueryPlan Schema

```json
{
  "intent": "formula_explanation",
  "paper_scope": ["paper_33eb859882132fd0"],
  "required_evidence_types": ["equation", "text"],
  "preferred_sections": ["Method"],
  "retrieval_mode": "reranked",
  "use_graph": false,
  "use_vlm": false,
  "use_judge": true,
  "confidence": 0.92,
  "needs_clarification": false
}
```

### 4.4 首批意图

| 意图 | 例子 | 检索/执行策略 |
|---|---|---|
| `definition` | “Q-Former 是什么？” | 单篇或全库文本检索；高置信度时可跳过精排 |
| `mechanism` | “如何连接视觉编码器和 LLM？” | Method 优先；文本、图注；Hybrid + Rerank + Judge |
| `motivation` | “为什么冻结 image encoder？” | Introduction、Method、Ablation/Discussion 优先 |
| `experiment` | “VQAv2 上表现如何？” | Table、Experiment；保留指标和设置上下文 |
| `comparison` | “BLIP-2 和 Flamingo 的区别？” | 多论文范围；进入受限调研计划或并行证据检索 |
| `attribution` | “这个结论来自哪一页？” | 精确 Chunk / 页码定位，不生成扩展结论 |
| `formula_explanation` | “式(3)是什么意思？” | equation + 前后文本上下文 |
| `figure_explanation` | “图 2 表达什么？” | figure caption + 邻近段落；必要时调用视觉工具 |
| `multi_hop` | “哪些方法使用冻结视觉编码器？” | 图谱实体链接 → 候选 Chunk → 证据包 |
| `follow_up` | “它为什么这样设计？” | Memory 实体解析；低置信度时澄清 |

### 4.5 验收指标

- 建立不少于 60 条、覆盖上述 10 类意图的 `routing_golden.v1.json`。
- 意图准确率 ≥ 0.85；证据类型选择 Macro-F1 ≥ 0.80。
- 模糊追问的错误上下文继承率 ≤ 5%；需要澄清的问题澄清召回率 ≥ 0.90。
- 对 `definition` 等简单问题，相比固定全链路平均延迟下降 ≥ 20%，且引用完整性不得下降。
- 所有路由决策写入 Trace：`intent`、`confidence`、`policy`、`fallback_reason`。

## 5. P0：Evidence Planner 与证据合并

### 5.1 目标

让系统先明确“回答该问题需要哪些证据”，再检索，而非由 Answer Agent 一边检索一边自由发挥。

### 5.2 输入与输出

```json
{
  "sub_questions": [
    "Q-Former 的结构是什么？",
    "它如何与冻结图像编码器交互？",
    "它如何将视觉信息接入冻结语言模型？"
  ],
  "required_sections": ["Method"],
  "required_evidence_types": ["text", "figure_caption"],
  "minimum_evidence_count": 3
}
```

Planner 只输出结构化计划，不得回答论文事实；计划经 Schema 校验后，由受控工具网关执行并行检索。

### 5.3 处理链路

```text
Evidence Planner
  → 子问题并行检索
  → 论文范围/类型/章节过滤
  → 去重（Chunk ID、文本相似、同页重复）
  → Reranker
  → Evidence Pack
```

### 5.4 验收指标

- 在机制、公式、图表和多论文问题子集上，Evidence Recall@5 相比 P0 前基线提升 ≥ 5 个百分点，或明确报告未提升的原因。
- Evidence Pack 的重复 Chunk 比例 ≤ 10%。
- 每个非拒答答案的关键 Claim 均可映射到至少一个 Evidence ID。
- Planner 输出必须通过 JSON Schema；非法计划、未知工具、未知 `paper_id` 拒绝执行。

## 6. P0：Evidence-aware Memory

### 6.1 目标

将记忆从“保存自然语言历史”升级为“保存带证据来源的会话状态”，既支持指代消解，又不让未验证摘要成为论文事实。

### 6.2 会话摘要最小结构

```json
{
  "active_paper_id": "paper_6bc5d399d6127a64",
  "active_topic": "Q-Former",
  "user_goal": "理解架构与设计动机",
  "resolved_entities": {
    "它": "Q-Former",
    "冻结模型": "frozen language model"
  },
  "findings": [
    {
      "summary": "Q-Former 使用查询向量提取视觉信息并向 LLM 提供软视觉提示。",
      "evidence_chunk_ids": ["child_93ef7db582e33f341708e079"],
      "paper_id": "paper_6bc5d399d6127a64"
    }
  ]
}
```

### 6.3 置信度与澄清

- 每个指代解析结果保存 `resolved_subject`、`resolved_reference`、`confidence`。
- 当 `confidence < threshold`，或候选实体跨论文冲突时，返回澄清问题，禁止直接继承上下文。
- 摘要压缩只在滑动窗口溢出后触发；摘要保留证据 ID、论文 ID、未解决问题和用户目标。

### 6.4 验收指标

- 新增不少于 20 条“指代消解 / 上下文漂移”标注案例。
- 上下文约束正确率 ≥ 0.90，错误继承率 ≤ 5%。
- 摘要中 `findings` 的证据 ID 可解析率为 100%。
- 记忆服务不可用时，系统降级但不泄露跨用户会话内容。

## 7. P1：诊断驱动的质量修复闭环

### 7.1 当前问题

Judge 发现 Claim 不受支持后，简单的“带反馈重写”无法区分：是引用错位、证据不足、问题过宽、指代不清，还是论文确实未提及。

### 7.2 Judge 诊断输出

```json
{
  "status": "failed",
  "failure_type": "insufficient_evidence",
  "unsupported_claims": ["某结论"],
  "recommended_action": "retrieve_more",
  "suggested_query": "Q-Former computational cost training efficiency"
}
```

### 7.3 故障—动作映射

| 失败类型 | 动作 |
|---|---|
| `citation_misalignment` | 保留证据，定向改写 Claim / 引用 |
| `insufficient_evidence` | 扩大 candidate-k、改写查询、重新检索和精排 |
| `scope_too_broad` | 拆分问题或请用户限定范围 |
| `ambiguous_reference` | 进入 Context Guard 澄清 |
| `not_in_corpus` | 明确拒答，不以通识补全 |
| `repeated_failure` | 达到预算后安全拒答 |

### 7.4 验收指标

- 每条失败 Trace 有 `failure_type` 与 `recommended_action`。
- “证据不足”案例经过一次 `retrieve_more` 后，证据召回或语义支持率提升 ≥ 10 个百分点；若无提升，应安全拒答。
- 不超过 2 次修复尝试；平均额外延迟、GPU 时间和失败率进入监控。
- 修复后不得降低 Citation Validity。

## 8. P1：Monitor、可观测性与发布门禁

### 8.1 每请求 Trace 字段

```json
{
  "trace_id": "tr_xxx",
  "query_sha256": "...",
  "intent": "mechanism",
  "retrieval_mode": "hybrid_rerank",
  "retrieved_chunk_ids": ["..."],
  "retrieval_latency_ms": 420,
  "generation_latency_ms": 2310,
  "judge_latency_ms": 850,
  "citation_valid": true,
  "unsupported_claim_count": 0,
  "answer_attempts": 1,
  "abstained": false
}
```

Trace 不保存原始问题或论文正文，只保存哈希、ID、聚合指标和脱敏错误类型。

### 8.2 监控指标

| 层级 | 指标 |
|---|---|
| 检索 | 无结果率、BM25/Dense 重叠率、Rerank 前后排名变化、重复 Chunk 率、Evidence 数量 |
| 生成 | Claim 数、引用覆盖率、无引用 Claim 率、重写次数、拒答率 |
| Judge | Unsupported Claim 率、Judge 失败原因、重试率、语义审查覆盖率 |
| 系统 | P50/P95 延迟、Embedding/Rerank/LLM 分段耗时、超时率、降级率、GPU 显存 |
| 计划 | 草案创建数、确认执行率、子任务成功率、计划完成率 |

### 8.3 发布门禁

候选配置变更（如 RRF 权重、candidate-k、路由阈值、Prompt）必须：

1. 在固定 Golden Set 回测；
2. 与当前基线比较；
3. 不降低 Citation Validity 与安全拒答指标；
4. 由人工确认后再更新配置。

禁止根据单次线上请求自动调整路由权重或安全阈值。

## 9. P1：Answer Golden Set 与六维评测

### 9.1 数据集定义

建立 `answer_golden.v1.json`：10 篇论文、30～50 个高质量问题。每题至少包括：

```json
{
  "case_id": "blip2_qformer_bridge_answer",
  "query": "How does Q-Former bridge ...?",
  "reference_answer": "...",
  "required_claims": ["..."],
  "required_evidence_chunk_ids": ["child_..."],
  "forbidden_unsupported_claims": ["..."],
  "should_abstain": false
}
```

### 9.2 六维指标

| 维度 | 指标 | 判定方式 |
|---|---|---|
| Retrieval | Hit、Recall、MRR、nDCG | 人工标注相关 Chunk |
| Citation Validity | 引用 ID 存在率 | 确定性规则 |
| Citation Completeness | 关键 Claim 引用覆盖率 | 确定性规则 + 标注 |
| Citation Correctness | 证据支持 Claim 的比例 | 独立 Judge + 人工抽检 |
| Answer Quality | 正确性、完整性、相关性 | Judge 与参考答案 |
| Abstention Quality | 应拒答时的正确拒答率 | 标注集确定性比较 |

### 9.3 验收指标

- Answer Golden Set ≥ 30 条，至少包含公式、表格、图示、跨论文、模糊追问和应拒答问题。
- 所有实验输出结构化 JSON，并可由报告脚本汇总。
- 每次配置变更生成与基线的指标差异报告。

## 10. P2：标准化工具接口与 MCP

### 10.1 Core Tool Registry

先完成内部稳定工具接口，再暴露 MCP：

```text
search_chunks
search_tables
search_figures
get_formula_context
get_section
compare_papers
query_paper_graph
validate_citations
judge_claim_support
inspect_experiment_result
```

每个工具必须有：输入 Schema、输出 Schema、权限 Scope、超时、重试、资源预算、审计 Trace。

### 10.2 MCP Server

当内部接口稳定并具备测试后，提供 MCP Server 适配层，使 Cursor、Claude Desktop 等外部 Agent 能调用论文解析、检索、图谱与评测能力。

验收条件：

- MCP 仅暴露只读查询和显式确认的受控写操作。
- 外部调用与 Web 调用复用同一工具网关与权限策略。
- 至少完成一个 Claude Desktop 或 Cursor 集成演示与集成测试。

## 11. 执行优先级

### 第一优先级：现在做

1. Query Router 与动态检索策略。
2. 表格、公式、图注等 Evidence Type Routing。
3. Evidence Planner、并行检索与去重。
4. Evidence-aware Memory Summary。
5. Trace 指标补齐与 Answer Golden Set。

### 第二优先级：完成上述数据基线后做

1. Judge 失败类型与诊断驱动修复。
2. 关系/多跳问题的 Neo4j Graph Retriever。
3. 多论文并行检索与受限综合。
4. 配置回测和人工发布门禁。

### 第三优先级：接口和产品化增强

1. MCP Server。
2. 多模型成本/延迟调度。
3. 图谱 Workspace 版本治理深化。
4. 基于稳定日志的策略优化研究。

## 12. 简历可陈述的最终成果

完成本路线后，项目可以准确描述为：

> 设计并实现 Evidence-Grounded Research Copilot：基于 LangGraph 构建意图与证据类型路由、混合检索与精排、结构化证据规划、引用约束生成和独立语义审查；通过诊断驱动的再检索/澄清/拒答闭环降低无证据回答风险。构建 Paper KG、会话记忆隔离、可观测 Trace 与 Answer Golden Set 回测体系，并将稳定能力以受控工具接口和 MCP 对外提供。

任何指标均应以实际评测报告为准；未完成的模块不得写成已上线能力。

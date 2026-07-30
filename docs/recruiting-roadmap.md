# Evidence-Grounded Research Copilot：秋招竞争力与演进路线

## 1. 项目定位

**Evidence-Grounded Research Copilot** 是一个面向算法研发团队的可追溯论文调研与知识沉淀系统。

它解决的不是“能不能聊天”，而是研发人员在调研论文时的三个实际问题：

1. 论文内容很长，方法、实验、公式和局限分散，定位证据成本高；
2. 大模型总结可能把不同论文、章节甚至常识混在一起，难以核验；
3. 团队成员各自读过的论文、问题和结论难以沉淀为可复用知识。

系统的核心承诺是：**回答中的每个事实性结论都应能回到论文的 Chunk、章节和页码；证据不足时应重写或拒答。**

```text
PDF 导入
  → 结构化解析 / 分块 / 索引
  → 检索、图谱与上下文消歧
  → Research Agent 生成带引用回答
  → Judge Agent 独立语义审查
  → 输出、拒答或重写
  → 用户确认后沉淀到个人论文图谱分支
```

## 2. 当前项目竞争力

### 2.1 已实现能力

| 能力方向 | 已实现内容 | 对应 AI 应用研发能力 |
|---|---|---|
| 文档理解 | MinerU 解析 PDF；统一为 `paper.json`；保留页码、bbox、章节、公式、图表等结构信息 | 非结构化数据治理、知识构建 |
| 检索系统 | BM25、BGE-M3 Dense Retrieval、RRF Hybrid、BGE Cross-Encoder Reranker | RAG 召回、排序优化、模型选型 |
| 离线评测 | 人工标注 Golden Set；Hit@K、Recall@K、MRR、nDCG@K；消融对比 | 自动化评测、Case 分析、回测 |
| 可信回答 | Claim-Evidence 映射、确定性引用完整性校验、引用页码与 Chunk 展示 | 幻觉控制、可解释 AI |
| 双 Agent | Research Agent 生成回答；独立 GLM Judge Agent 审查每条 Claim 的证据支持性 | 多 Agent 职责解耦、反思纠错闭环 |
| 上下文工程 | Context Guard、实体消歧、显式论文优先、会话当前论文约束 | 意图识别、上下文注入、对话防跑偏 |
| 记忆与图谱 | Session / Episodic / Summary / Profile 分层记忆；Paper KG 与用户记忆隔离；workspace 分支 | 记忆管理、知识治理、多用户隔离基础 |
| 工具编排 | 工具依赖、并行批次、超时、重试、冲突隔离；真实检索链路接入调度 | 工具调用、Agent 调度、稳定性设计 |
| 服务化 | FastAPI、Streamlit、PDF 上传任务、独立 Worker、Neo4j 可选物化 | API 服务、异步任务、产品化 Demo |

### 2.2 已获得的量化结果

在 10 篇代表性 VLM 论文、48 条人工标注检索问题上：

| 检索方案 | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6250 | 1.0000 | 0.1679 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3 Dense | 0.7083 | 1.0000 | 0.2030 | 0.6005 | 0.8420 | 0.6377 |
| BM25 + BGE-M3 RRF | **0.7917** | 1.0000 | **0.2252** | 0.6606 | **0.8889** | 0.6906 |
| RRF + Cross-Encoder Rerank | 0.7500 | 1.0000 | 0.2099 | **0.6944** | 0.8611 | **0.6943** |

在 12 个已生成回答、37 条 Claim 的回答质量评测中：

| 指标 | 结果 |
|---|---:|
| 引用完整性通过率 | 100% |
| 语义审查覆盖率 | 100% |
| 语义支持率 | 100% |
| 未支持 Claim 比例 | 0% |
| 完全受支持回答比例 | 83.3% |

这些结果表明：RRF 更擅长提高首条命中和整体排序稳定性；Cross-Encoder 更擅长在 Top-5 中找全多个相关证据。两者应按业务目标组合，而不是只宣传某一个“最强”方案。

### 2.3 对秋招 JD 的价值

项目已经不是单一“调用大模型”的 Demo，而是能够举证以下工程判断：

- 对模糊问题使用 Context Guard 和实体消歧，而不是直接扩大检索范围；
- 对事实回答采用确定性引用校验与独立 Judge 的双重约束；
- 对昂贵或慢速模型任务使用异步导入、超时、重试和降级；
- 对 Paper KG、用户画像、会话状态和生成答案进行数据边界隔离；
- 用人工 Golden Set、检索指标与 Claim 级语义指标驱动迭代，而非凭主观感受调 Prompt。

## 3. 当前架构与边界

```text
Web UI (Streamlit)
       │
FastAPI API
  ├─ Context Guard / Entity Resolver
  ├─ Tool Orchestrator
  ├─ Retrieval: BM25 + Dense + RRF + Rerank
  ├─ Research Agent (Qwen)
  ├─ Citation Validator (deterministic)
  ├─ Judge Agent (GLM, optional independent service)
  ├─ Agent Memory
  └─ Paper KG / Graph Workspace
       │
Ingestion Worker
  ├─ MinerU
  ├─ Normalize / Chunk
  └─ Dense Index
```

### 3.1 双 Agent 的定义

当前系统是合理的最小双 Agent 架构，而非为了形式拆分的多 Agent：

- **Research Agent**：基于检索证据生成回答、Claim 和引用；
- **Judge Agent**：独立评估每条 Claim 是否被其引用 Evidence 支持，失败时给出结构化反馈；
- **确定性组件**：检索、引用 ID 校验、任务状态、权限、超时与降级不交由 LLM 自由决定。

这形成“生成—审查—重写/拒答”闭环。Judge 不负责生成内容，Research Agent 不负责自证正确性，从而降低同模型自我确认偏差。

### 3.2 工具层的当前状态

项目已有检索、图谱查询、记忆读写、任务状态、语义审查等工具能力，也已有超时、依赖和重试调度骨架。

下一阶段的重点不是增加更多工具，而是将真实工具统一为产品级契约：

```text
ToolSpec
  name / 输入输出 Schema / 权限 / 超时 / 重试 / 资源标签 / 是否幂等

ToolResult
  trace_id / 状态 / 耗时 / 尝试次数 / 降级信息 / 错误 / 结构化输出
```

## 4. 下一阶段项目设计

优先目标：从“功能丰富的论文 Agent”演进为“可观测、可评测、可恢复的研发知识助手”。

### P1：全链路可观测性与归因

为每次问答和导入任务分配 `trace_id`，记录：

- 用户请求：query、论文范围、消歧结果、拒答原因；
- 检索：各召回器候选数、Top-K、排序结果、耗时；
- 模型：模型名、输入输出 token、延迟、重试次数；
- 工具：依赖、超时、降级、错误类型；
- 任务：MinerU、分块、索引各阶段耗时与成功率；
- 质量：引用校验、Judge Verdict、用户是否采纳回答。

交付物：结构化 Trace、单请求详情页、P50/P95 延迟与失败率看板。

### P2：异步任务系统与稳定性

保留现有 JSON 任务状态机作为逻辑模型，升级运行载体：

- Redis 负责队列、锁、限流与短期状态；
- 优先使用 RQ（低学习成本），再理解 Celery 的复杂场景；
- `ingestion`、`answer`、`judge` 三类队列隔离资源；
- 以 PDF SHA256 作为幂等键，避免重复解析和重复向量化；
- 重试策略区分网络错误、模型暂时不可用、参数错误；
- 降级策略：Rerank 超时回退 Hybrid；Judge 不可用时保留 Citation Validation 并显式标识；解析失败保留原 PDF 和任务日志。

### P3：安全与多用户数据边界

论文 OCR 文本属于不可信输入，必须避免它影响系统指令或触发越权工具：

- 分离 System Prompt、用户问题、检索证据，并明确证据不是执行指令；
- 工具白名单和 Pydantic 输入 Schema；
- PDF 类型、大小、页数、文件名、SHA256 校验；
- workspace ownership 校验，禁止跨用户读取记忆和图谱分支；
- 增加 Prompt Injection、越权访问、超长上下文等红队测试集。

### P4：受治理的真实工具编排

将下列能力统一注册，而非在路由和脚本中分散直接调用：

- `resolve_context`
- `retrieve_evidence`
- `query_paper_graph`
- `inspect_ingestion_task`
- `promote_paper_to_workspace`
- `generate_grounded_answer`
- `validate_citations`
- `semantic_judge`

调度原则：

- 稀疏检索、稠密检索、图谱查询可并行；
- 回答生成必须依赖证据包；
- 语义审查必须依赖 Citation Validation；
- 写 workspace 与读取 workspace 需按一致性要求串行；
- 不让 LLM 自由执行任意 Shell、数据库或文件系统操作。

### P5：业务指标与回归评测

保留当前人工 Golden Set 作为主评测，同时引入 RAGAS 作为自动回归补充：

- 主评测：Hit@K、Recall@K、MRR、nDCG@K、人工答案引用标注；
- 自动评测：RAGAS Faithfulness、Context Precision、Context Recall；
- 回答可信度：Citation Pass Rate、Semantic Support Rate、Unsupported Claim Rate、拒答准确率；
- 系统指标：P50/P95 延迟、任务成功率、工具超时率、GPU 时间、单请求成本；
- 产品指标：证据展开率、回答采纳率、论文入个人分支比例。

## 5. 开发优先级与非目标

### 建议顺序

1. 完成网页导入、Worker、个人 workspace 确认提交的闭环；
2. P1 可观测性；
3. P2 Redis + 异步队列；
4. P3 安全与红队测试；
5. P4 真实工具注册与统一调度；
6. P5 RAGAS 与线上质量回归；
7. 最后再考虑 Neo4j 的在线查询优化、多用户鉴权、团队共享和部署。

### 当前不应优先做的事情

- 为了标签强行增加很多互相聊天的 Agent；
- 同时引入 LangChain、LangGraph、AutoGen 等多个框架；
- 只因“图数据库很酷”而迁移全部 JSONL 图谱；
- 未有用户/任务规模前就设计复杂微服务；
- 继续盲目扩大论文数量，却没有增加高质量标注与失败 Case。

## 6. 六周学习路线

| 周次 | 主题 | 目标交付物 |
|---|---|---|
| 第 1 周 | FastAPI、文件上传、状态机、进程管理 | 完成论文上传—Worker—个人分支闭环 |
| 第 2 周 | OpenTelemetry、结构化日志、Langfuse/Phoenix | 可查询的单请求 Trace 和延迟看板 |
| 第 3 周 | Redis、RQ、幂等、重试、死信任务 | 可恢复的异步导入队列 |
| 第 4 周 | Prompt Injection、鉴权、限流、红队测试 | 安全测试集与降级策略报告 |
| 第 5 周 | vLLM、KV Cache、continuous batching、SSE | 一个 OpenAI-compatible 推理服务及延迟对比 |
| 第 6 周 | 项目表达、架构图、Demo、失败案例复盘 | README、实验报告、3 分钟 Demo、面试题库 |

学习原则：每学习一个主题，都必须在项目中留下可运行代码、测试、指标或文档，而不是只看教程。

## 7. 简历表述草案

> 构建面向算法研发团队的可追溯论文调研 Agent，完成 PDF 结构化解析、BM25+BGE-M3 混合检索与 Cross-Encoder 精排；在 10 篇 VLM 论文、48 条人工标注问题上，Hybrid Hit@1 达 0.7917，精排 Recall@5 达 0.6944、nDCG@5 达 0.6943。

> 设计 Research–Judge 双 Agent 可信回答闭环：Research Agent 生成 Claim-Citation 映射，独立 GLM Judge 审查 Claim 与证据的语义支持关系；在 12 个回答、37 条 Claim 上实现引用完整性 100%、语义支持率 100%、未支持 Claim 比例 0%。

> 实现 Context Guard、分层 Agent Memory、个人 Paper KG workspace 分支与异步 PDF 导入任务；通过工具超时、重试、依赖调度与降级策略提升系统可恢复性。

## 8. 面试中应主动说明的边界

- 当前评测集规模有限，48 条问题适合验证迭代方向，不应宣称为通用学术 QA Benchmark；
- 100% 语义支持率来自已审查样本，依赖于当前 Judge 和人工设计的评测范围；
- JSONL 图谱是事实源，Neo4j 是可选物化视图，避免引入数据库后失去版本可追溯性；
- 当前任务队列是单机 MVP，Redis/RQ 是下一阶段为并发、恢复和多用户做的演进，而不是虚构“已上线分布式系统”。

诚实说明边界，配合可复现指标和失败案例，比夸大“企业级、多 Agent、生产可用”更能体现工程判断力。

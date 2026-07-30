# 读论文 Agent 系统设计与执行方案

## 1. 目标与边界

本项目定位为**证据驱动的论文研究工作台**，支持：

1. 论文上传、解析、索引、版本与工作区管理；
2. 单篇论文讲解，包括方法、公式、图表、实验和局限；
3. 论文库内的技术发现与多论文结构化对比；
4. 基于已选论文的、可追溯的文献综述写作；
5. 经用户确认后的联网候选文献发现。

核心约束：

```text
关键结论 -> Claim -> Evidence Chunk -> 论文 / 章节 / 页码
```

系统不得使用论文库中的其他内容，补全当前上传论文没有明确给出的事实；也不得把“当前论文库未发现”表述为“全领域无人提出”。

## 2. 总体架构

```mermaid
flowchart TD
    U[用户输入或模式选择] --> R[总控 Router]
    R --> M[论文库管理 Agent]
    R --> A[研读与对比 Agent]
    R --> W[综述写作 Agent]

    M --> S[共享能力层]
    A --> S
    W --> S

    S[CorpusScope / 检索 / 图谱 / Context Manager / Citation Guard / Memory / Hooks / 权限网关 / Trace]
```

对用户只暴露三个专业 Agent：

| Agent | 用户能力 | 编排方式 |
| --- | --- | --- |
| 论文库管理 Agent | 上传、解析、索引、图谱、入库与版本 | 固定任务工作流 + 异步 Worker |
| 研读与对比 Agent | 单篇讲解、公式图表解释、库内技术发现、多论文对比 | LangGraph 子图 |
| 综述写作 Agent | 规划、证据矩阵、章节写作、反思、审计与导出 | 嵌套 LangGraph 子图 |

总控 Router 是调度器，不是额外的业务 Agent。

## 3. 技术选型

### 3.1 LangGraph、LangChain 与固定工作流

- **LangGraph**：用于需要状态、条件分支、工具调用、反思循环、人工确认、checkpoint 或跨 Agent handoff 的业务子图。
- **固定工作流 / Worker**：用于 PDF 解析、分块、索引等确定性、长耗时任务；不将其伪装为自由 Agent。
- **LangChain**：可作为模型适配、检索器、文档加载和提示模板等底层组件，不使用其自由 Agent Executor 承担核心业务编排。

选择 LangGraph 的原因是论文研究任务需要可审计状态、受控工具调用、可恢复执行、有限循环和人审中断；这些要求高于“让模型自主循环调用工具”的便利性。

### 3.2 保持现有模型策略

| 角色 | 当前模型 / 组件 |
| --- | --- |
| 论文问答与图文理解 | Qwen3-VL-8B-Instruct |
| Claim 语义审计 | GLM-4.1V-9B-Thinking |
| 检索嵌入 | BGE-M3 |
| 重排 | BGE Reranker v2-m3 |
| 论文解析 | MinerU |

Qwen 负责生成与多模态理解，GLM 作为独立 Judge，避免同一模型既生成又自审造成的偏差。

## 4. CorpusScope：论文范围隔离

`CorpusScope` 必须成为请求、检索、生成、审计、Trace 和 Handoff 的显式字段。

| Scope | 含义 |
| --- | --- |
| `temporary_workspace` | 新上传、尚未确认入库的论文 |
| `active_paper_only` | 仅当前阅读论文 |
| `selected_papers` | 仅用户明确选中的论文 |
| `library_only` | 已索引论文库 |
| `web_expansion` | 经确认的联网候选文献发现 |

默认流程：

```text
上传论文 -> temporary_workspace -> active_paper_only
用户要求库内技术发现 -> library_only
用户确认联网扩展 -> web_expansion
```

每个回答和综述章节均显示其证据范围。

## 5. 总控 Router

Router 的优先级为：用户手选模式 > 规则路由 > 低置信度时的结构化模型分类 > 用户澄清。

Router 可产生有依赖关系的多 Agent 调度，但不能直接执行检索、联网、文件或数据库写入。例如：

```json
{
  "dispatches": [
    {
      "agent": "reading_compare",
      "mode": "active_paper_explain",
      "corpus_scope": "active_paper_only",
      "depends_on": []
    },
    {
      "agent": "reading_compare",
      "mode": "library_related_work",
      "corpus_scope": "library_only",
      "depends_on": ["active_paper_explain"]
    }
  ],
  "needs_confirmation": false
}
```

## 6. 论文库管理 Agent 工作流

```text
上传论文
-> 文件、格式、大小和权限校验
-> 创建临时阅读工作区
-> PDF 解析 / OCR / 版面恢复
-> 章节、文本、表格、公式、图片分块
-> 解析质量检查
-> 生成元数据、章节卡和证据卡
-> 稀疏索引、向量索引、概念图谱构建
-> 用户预览
-> 显式确认后提升至论文库或 Workspace
```

每篇论文离线构建四层资产：

```text
Document Map -> Section Card -> Evidence Card -> 原始 Chunk / 图 / 表 / 公式
```

前三层用于导航和筛选；最终 Claim 只能引用原始 Chunk 及其页码锚点。

## 7. 研读与对比 Agent 工作流

```text
Scope Guard / Context Guard
-> 意图与模式识别
-> Evidence Planner Agent
-> Query Rewriter
-> 本地检索 / 图谱查询 / 可选联网候选发现
-> Evidence Merge、去重、重排
-> 单篇讲解 / 技术发现 / 比较矩阵
-> Citation Guard
-> Semantic Judge
-> Reflection：补检索、局部改写、澄清或安全拒答
```

### 7.1 模式

1. `active_paper_explain`：仅当前论文，解释方法、公式、图表、实验和局限。
2. `library_related_work`：将用户想法映射为相同、部分相似、术语相近或未发现四类关系。
3. `multi_paper_compare`：按研究问题、架构、训练目标、数据、指标、成本与局限构建证据矩阵。

比较时每篇论文、每个比较维度只保留 1--2 条高质量证据，禁止将多篇全文送入模型上下文。

## 8. 综述写作 Agent 工作流

```mermaid
flowchart TD
    I[主题、范围、模板、篇幅] --> P[ReviewPlanner Agent]
    P --> V[Plan Validator]
    V --> H[用户确认大纲与语料范围]
    H --> E[Evidence Executor]
    E --> L[本地检索]
    E --> G[图谱查询]
    E --> N[联网候选发现，可选]
    N --> C[用户确认候选论文]
    C --> E
    L --> M[Evidence Matrix Builder]
    G --> M
    M --> W[Section Writer Agent]
    W --> Q[Citation / Coverage Validator]
    Q --> R[Reflection Agent]
    R -->|补证据| E
    R -->|重规划| P
    R -->|通过| O[用户审阅与导出]
```

内部节点职责：

| 节点 | 类型 | 职责 |
| --- | --- | --- |
| `ReviewPlanner` | Agent | 生成大纲、比较维度、依赖、证据需求和预算 |
| `PlanValidator` | 固定 | 校验 Schema、范围、预算和联网授权 |
| `EvidenceExecutor` | 固定 | 执行受白名单约束的检索和图谱工具 |
| `EvidenceMatrixBuilder` | Agent + 校验 | 构建章节—观点—论文—证据矩阵 |
| `SectionWriter` | Agent | 只基于当前章节证据起草 |
| `CitationGuard` | 固定 | 校验 Claim 与证据映射 |
| `SemanticJudge` | 独立模型 | 审计语义支持关系 |
| `ReflectionAgent` | Agent | 诊断问题并生成受限修复动作 |
| `HumanReview` | 人工节点 | 确认大纲、候选论文、最终稿与导出 |

Reflection 仅允许：`retrieve_more`、`rewrite_section`、`split_scope`、`ask_user_clarification`、`mark_insufficient_evidence`、`safe_refusal`。每章节最多循环 1--2 次。

## 9. Query Rewrite 的位置

问题改写位于“任务范围和证据需求明确之后、检索之前”：

```text
原始问题
-> 输入校验
-> 会话与指代消解
-> Router
-> Evidence Planner
-> Query Rewriter
-> RAG / 图谱 / Web 检索
-> Evidence Pack
-> 生成与审计
```

改写器输出多个受控检索查询，不改写最终要回答的原问题。它的输入包括原问题、已解析实体、`CorpusScope`、任务模式和证据类型要求。明确论文 ID、公式编号、图表编号或术语时可跳过 LLM 改写。

## 10. Context Manager

原则：**全文保留在存储层，模型只接收完成当前节点所需的最小可追溯证据包。**

上下文分为四层：

1. 固定运行上下文：系统规则、权限、Scope、Schema、预算；
2. 任务上下文：当前 Agent、节点、计划、章节和待处理任务；
3. 会话上下文：最近对话、压缩摘要、已确认实体、未解决问题；
4. 证据上下文：当前任务选择的原始 Chunk、表格、公式、图像和页码。

建议的初始总预算为 24K--32K token：

```text
系统规则与安全：2K
任务状态：2K
会话记忆：2K
论文证据：12K--16K
当前草稿：4K
输出预留：2K--4K
```

综述按章节隔离：仅加载全局大纲摘要、当前章节目标、当前章节证据矩阵、当前章节原始证据和必要的过渡摘要。摘要仅用于导航，不能替代原始证据。

## 11. Hooks 与 Handoff

### 11.1 Hooks

| Hook | 职责 |
| --- | --- |
| `before_route` | 身份、输入、Scope 和速率检查 |
| `before_model` | Context Budget、Prompt 边界和输出 Schema |
| `before_tool` | Scope、参数、网络/写入审批和预算校验 |
| `after_tool` | 结果 Schema、脱敏和来源记录 |
| `after_model` | JSON、Claim 和引用要求校验 |
| `on_transition` | Trace、checkpoint、成本和错误记录 |

Hooks 是固定治理代码，不允许模型修改或绕过。

### 11.2 Handoff

Handoff 只用于跨 Agent 边界：论文库管理到研读与对比、研读与对比到综述写作、综述写作到论文库管理。统一使用 `HandoffPacket`，其中只传递 Scope、论文 ID、Artifact 引用、用户目标、已授予权限与确认状态；不传全文或完整聊天历史。

## 12. 权限与沙箱

| Agent | 默认权限 | 需显式确认的权限 |
| --- | --- | --- |
| Router | `session:read`、`route:plan` | 无 |
| 论文库管理 | `file:upload`、`ingestion:run`、`paper:read` | `library:write`、`workspace:write` |
| 研读与对比 | `paper:read`、`graph:read`、`memory:read` | `web:discover` |
| 综述写作 | `paper:read`、`graph:read`、`artifact:write` | `web:discover`、`export:write` |
| Judge | `evidence:read` | 无写入权限 |

必须确认的动作包括扩大检索范围、联网搜索、下载外部论文、写入共享论文库、导出文档和覆盖已有草稿。

PDF 解析在低权限 Worker/容器中执行；输入目录只读，输出仅写入任务临时目录；默认禁网；模型不可调用任意 Shell、路径或数据库语句，亦不可读取密钥。论文和网页正文均视为不可信证据并以明确边界注入 Prompt。

## 13. 可观测性与评测

每次运行记录：Agent、节点、Scope、检索查询数、选中/丢弃证据数、token 预算、工具耗时、重试、反思动作、引用结果和拒答原因。Trace 不保存完整论文正文或密钥。

新增评测集与门禁：

1. 当前论文范围泄漏测试；
2. Router 单/多 Agent 调度正确率；
3. Query Rewrite 的检索增益及意图漂移率；
4. 多论文比较的 Evidence Recall、Citation Validity、矩阵覆盖率；
5. 综述的章节引用覆盖率、证据矩阵覆盖率、人工质量评测；
6. 成本、延迟、上下文预算、失败率和拒答率监控。


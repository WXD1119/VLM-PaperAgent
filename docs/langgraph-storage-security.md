# LangGraph、三库存储与安全边界

## 迁移目标

将原有确定性的 Research-Judge 工作流迁移为 LangGraph `StateGraph`，但不让 LLM 获得任意工具、Shell 或数据库写入权限。LangGraph 在本项目中是状态机运行时，不替代领域代码：节点复用现有检索器、`AnswerAgent`、`CitationValidator`、`SemanticCitationJudge` 与评测数据。

图的流程如下：

```text
START
  -> resolve_context
  ├-> clarification -> END
  └-> retrieve_evidence -> generate_answer -> validate_citations
       ├-> invalid -> refuse -> END
       └-> semantic_judge
            ├-> supported -> END
            ├-> unsupported and attempts remain -> generate_answer
            └-> attempts exhausted -> refuse -> END
```

LangGraph 的条件边由确定性代码控制。Judge 不支持某个 claim 时，图只会携带反馈重写一次或安全拒答，不会把失败答案直接作为可信结果输出。

## 存储职责

| 存储 | 保存内容 | 明确不保存 | 生命周期 |
|---|---|---|---|
| Redis | 当前 `SessionState`、最近 N 轮原始对话窗口、短期任务指针 | 长期画像、论文事实、完整 Paper KG | TTL，默认 24 小时 |
| MySQL | 用户画像、Episodic Event、压缩 Summary、可审计对话历史 | 论文 Chunk 向量、论文概念关系 | 持久化，按用户隔离 |
| Neo4j | Paper、Section、Chunk、Concept 及其证据关系 | 用户对话、用户偏好、生成答案、会话状态 | 持久化 Paper KG / Workspace 物化 |
| ChromaDB | 论文 Chunk 向量索引 | 用户记忆、权限信息 | 可重建索引 |
| JSON / JSONL artifacts | 原始 PDF、解析产物、回答工件、离线回放 | 在线会话真值 | 可审计与迁移回退 |

Neo4j 在本项目中是“论文语义知识记忆”，不是用户记忆库。论文图谱不能混入用户画像、对话原文或 Agent 自身生成内容；这也是分支论文图谱可复现、可共享的前提。

## 安全边界

### 身份与数据隔离

- 每个请求在网关层建立 `ActorContext(user_id, session_id, scopes)`。
- Redis Key 使用 `user_id + session_id` 命名空间；MySQL 所有记忆记录带 `user_id`。
- `workspace:write` 与 `paper:read` 是不同 Scope；模型不能通过文本为自己增加 Scope。
- 当前 `trusted-header` 模式仅适用于已经校验 JWT/OIDC、并会剥离外部同名请求头的反向代理。直接暴露 API 时必须先接入真正鉴权，不能把裸 HTTP Header 当身份凭证。

### Prompt Injection 防护

- 用户问题、系统指令与检索到的论文文本处于不同 Prompt 区域。
- OCR / Chunk 一律包装为 `UNTRUSTED_EVIDENCE`，只可作为事实来源。
- 证据中出现“忽略之前指令”“调用工具”等文字，只能被引用，不能改变状态图的工具调用路线。
- 可疑注入文本会记录安全信号，供审计和限流使用；不因包含安全术语就拒绝正常研究问题。

### 工具与写入边界

- LangGraph 条件边由确定性代码控制；第一版不允许 LLM 自由选择任意工具。
- 工具注册应声明输入/输出 Schema、超时、重试、资源与 `required_scopes`。
- 问答默认只写回答工件、Redis 会话和 MySQL 用户记忆；论文图谱写入仍需要用户确认后的 workspace promotion。
- 上传文件需要类型、大小、页数、SHA256 和路径校验；解析器、模型服务与 Web API 应分进程运行。

## 代码结构

```text
src/paper_agent/
  runtime/
    research_graph.py       # LangGraph Research-Judge 状态机
  security/
    boundaries.py           # Actor、Scope、输入安全信号
    evidence.py             # 不可信论文证据 Prompt 边界
  memory/
    backends.py             # Redis 与 MySQL 适配器
  graph/                    # 论文语义知识图谱与 workspace
  tools/                    # 超时、依赖、重试控制的工具层
```

## 本地启动依赖

```bash
docker compose up -d redis mysql neo4j
pip install -e '.[api,agent,storage,ui]'

export REDIS_URL='redis://:paper-agent-dev@127.0.0.1:6379/0'
export MYSQL_URL='mysql+pymysql://paper_agent:paper-agent-dev@127.0.0.1:3306/paper_agent'
export NEO4J_URI='bolt://127.0.0.1:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='paper-agent-dev'
export PAPER_AGENT_MEMORY_BACKEND='distributed'
```

远程服务器没有 Docker 时，不需要改代码：通过环境变量接入学校或云端提供的 Redis、MySQL、Neo4j，或者保留 `PAPER_AGENT_MEMORY_BACKEND=file` 作为离线回退。开发环境 Docker 密码仅用于本机，不可复用到线上。

## 验收

```bash
python -m pytest tests/test_langgraph_runtime.py tests/test_memory_backends.py tests/test_security_boundaries.py -v
python -m pytest -v
```

验收重点：

1. Context Guard 要求澄清时，LangGraph 不得调用检索或模型；
2. Judge 不支持 Claim 时，图必须重写或安全拒答；
3. Redis 不同用户或会话的 Key 不得互相读取；
4. MySQL 长期记忆查询必须带 `user_id`；
5. Neo4j 导出仍只包含论文内容；
6. 证据中的恶意指令只能被引用，不能改变工具调用路线。

LangGraph 的显式 State、节点和条件边设计见其官方 [Graph API 文档](https://docs.langchain.com/oss/python/langgraph/use-graph-api)。

## 运行追踪

每次 `/ask` 会返回 `trace_id`，并将脱敏后的轨迹追加到
`artifacts/traces/traces.jsonl`。Trace 记录以下信息：

- LangGraph 节点的执行耗时与成功/失败状态；
- 最终证据数、引用校验结果、Judge 结果、重试次数和拒答标记；
- 问题的 SHA256，而不是原始问题文本；
- 截断后的异常类型与摘要，而不是论文正文、密钥或模型完整输出。

查询接口：

```bash
curl http://127.0.0.1:8000/traces?limit=20
curl http://127.0.0.1:8000/traces/<trace_id>
```

Streamlit 的“运行追踪”页会将 Trace 展示为节点耗时表。Trace 查询按 `user_id`
过滤；生产环境仍需在 JWT/OIDC 网关后启用 `trusted-header` 模式，不能把裸 Header
视为真正身份认证。

## 质量闭环与端到端评测

Judge 发现不受支持的 Claim 时，LangGraph 不会盲目重写整段回答：它仅将失败 Claim
及其判定原因反馈给生成节点，要求保留已经支持的 Claim。达到最大尝试次数后，系统以
`semantic_support_failed` 拒答；确定性引用校验失败则以 `citation_integrity_failed` 拒答。

运行中的 API 可使用人工标注检索集进行端到端回测：

```bash
python scripts/evaluate_end_to_end.py \
  --golden evals/retrieval_golden.v2.json \
  --api-url http://127.0.0.1:8000 \
  --top-k 5 \
  --output artifacts/evals/end_to_end.v2.json
```

该报告包含 Evidence Hit@5、Evidence Recall@5、引用通过率、Judge 覆盖率/通过率、
重写率与安全拒答率。它与离线的 BM25/Dense/Hybrid/Reranker 评测互补：前者衡量真实
Agent 闭环，后者定位检索模块本身。

## 三库存储部署验收

启动 API 后先检查：

```bash
curl -s http://127.0.0.1:8000/health/storage | python -m json.tool
```

它只执行 Redis `PING`、MySQL `SELECT 1` 与 Neo4j 连接校验，不读取用户或论文数据。
当 `PAPER_AGENT_MEMORY_BACKEND=distributed` 时，Redis/MySQL 异常会记录 `memory_degraded`
Trace 并回退为文件会话或不写长期记忆；问答主链路仍可返回证据化回答。Neo4j 故障不会
污染用户记忆，因为它只承载论文图谱的可选物化视图。

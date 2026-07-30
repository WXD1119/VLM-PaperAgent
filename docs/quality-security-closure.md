# 可信 Agent 闭环：质量、评测、工具与三库存储

## 1. Judge 不是旁路评分，而是质量门

LangGraph 主路径为：

```text
检索 → 生成 Claim + Citation → 确定性引用校验 → 独立语义 Judge
                                                  ├─ 全部支持 → 输出
                                                  ├─ 有失败 Claim 且可重试 → 定向重写
                                                  └─ 重试耗尽 → 安全拒答
```

定向重写只将不通过 Claim 的编号、判定和原因反馈给生成器，要求保留已支持的 Claim。这样
避免“整段重写”覆盖正确内容并引入新的幻觉。

拒答类型：

| refusal_kind | 含义 |
|---|---|
| `needs_clarification` | 问题缺少必要上下文，不能安全检索。 |
| `citation_integrity_failed` | Claim 与 Evidence ID 的确定性约束不成立。 |
| `semantic_support_failed` | 独立 Judge 在重试后仍发现 Claim 不能由证据支持。 |
| `evidence_insufficient` | 检索证据不足，无法形成可追溯回答。 |

## 2. 两层评测

离线检索评测回答“召回器有没有找对证据”：Hit@K、Recall@K、MRR、nDCG@5。

端到端评测回答“一个完整 Agent 请求是否可靠”：

- Evidence Hit@5 / Recall@5；
- 引用完整性通过率；
- Judge 覆盖率与通过率；
- 重写率；
- 安全拒答率。

运行：

```bash
python scripts/evaluate_end_to_end.py \
  --golden evals/retrieval_golden.v2.json \
  --api-url http://127.0.0.1:8000 \
  --top-k 5 \
  --output artifacts/evals/end_to_end.v2.json
```

这项测试会真实调用模型。建议先用 `--limit 5` 做烟雾测试，确认稳定后再运行完整 48 条集。

## 3. 工具调度与安全

`AuthorizedToolGateway` 是 LangGraph 节点进入业务能力的唯一入口。它在调用前执行：

1. 工具白名单与 Scope 校验；
2. 输入/输出 Schema 元数据检查；
3. 超时、重试和资源冲突控制；
4. 失败后抛出明确异常，由状态图安全终止。

LLM 只生成回答和 Judge 结构化判断，不能提供任意工具名、Python 代码、数据库连接或 Scope。
`promote_paper_to_workspace` 需要 `workspace:write`，普通问答身份只拥有 `paper:read`。

## 4. 三库存储的职责与验收

| 存储 | 职责 | 安全要求 |
|---|---|---|
| Redis | 会话状态、最近对话窗口 | Key 包含 user_id 与 session_id，使用 TTL。 |
| MySQL | 用户画像、事件、摘要、可审计对话 | 每次读写带 user_id；禁止用 Paper KG 代替。 |
| Neo4j | 论文概念、章节、Chunk 与证据关系 | 只物化论文节点与边；禁止导入回答、用户或会话。 |

健康检查：

```bash
curl -s http://127.0.0.1:8000/health/storage | python -m json.tool
```

它只执行 Redis `PING`、MySQL `SELECT 1`、Neo4j 连接验证。Redis/MySQL 故障时，问答会
保留主回答，并在 Trace 中记录 `memory_degraded`；此时不应把会话连续性当作可靠能力。

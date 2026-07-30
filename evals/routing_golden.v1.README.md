# 路由标注集 v1（草案）

`routing_golden.v1.draft.json` 包含 60 条用于 Query Router 的人工复核草案，覆盖：定义、机制、动机、实验、比较、公式、表格、图示、多跳关系和多轮追问，各 6 条。

## 标注字段

- `intent`：预期问题意图。
- `paper_scope`：预期论文范围；使用论文短名，路由实现时再映射为真实 `paper_id`。
- `expected_evidence_types`：至少应检索的证据类型；`text`、`equation`、`table`、`figure`。
- `expected_use_graph`：是否建议进入图谱候选发现；即使为真，最终答案仍需回到论文 Chunk 引用。
- `expected_needs_clarification`：缺少论文、图表编号或会话实体时，系统应澄清而不是猜测。
- `session_context`：仅用于多轮追问案例；没有该字段表示无可信会话状态。

## 人工复核要求

1. 逐条检查意图是否符合问题真正需要的证据策略，而非仅检查关键词。
2. `paper_scope` 统一改为项目真实 `paper_id`，或在后续加载器中维护短名映射。
3. 当问题同时需要正文和图表/公式上下文时，保留多个 `expected_evidence_types`。
4. 对应澄清案例必须确保没有提供可用会话上下文；否则不应标为 `needs_clarification=true`。
5. 人工确认后复制为 `routing_golden.v1.json`，不要直接将草案用于正式回归门禁。

# 两层研究运行时

项目不把所有请求都当作开放式 ReAct 任务处理，而是区分两个运行层。

## 第一层：证据问答工作流

适用于单篇论文问题、明确概念解释和追问。LangGraph 固定执行：上下文消歧、检索、证据包构建、回答、引用完整性校验、独立语义审查，以及必要时的定向改写或安全拒答。

这层的特点是路径确定、可追踪，并且模型不能自行决定调用任意工具。

## 第二层：受限 Plan-and-Execute 调研

适用于“比较多篇论文的方法与训练策略”之类的复杂目标。

1. 用户明确输入调研目标和至少两篇 `paper_id`。
2. 系统生成可见草案：每篇论文一个 `paper_qa` 子任务，最后一个 `synthesize` 汇总任务。
3. 用户检查论文范围和任务列表，显式确认后才执行。未确认时不会调用模型、检索、图谱写入或消耗 GPU。
4. 每个 `paper_qa` 子任务仍使用第一层证据问答工作流，并保存对应 Trace 与证据 Chunk ID。
5. 汇总步骤只拼接已完成子任务的结论及其 Chunk 来源；它不是一条可被当成论文事实的新断言。

计划、子任务结果和汇总属于用户侧任务记录，保存到本地 JSON（开发回退）或 MySQL；绝不写入 Neo4j Paper KG。Neo4j 中只保留已解析论文的事实性内容与其关系。

## API

- `POST /research/plans`：创建草案。
- `GET /research/plans`、`GET /research/plans/{plan_id}`：按当前用户查看草案和执行结果。
- `POST /research/plans/{plan_id}/execute`：请求体必须包含 `{"confirm": true}` 才会执行。

生产环境应启用可信身份网关（`PAPER_AGENT_AUTH_MODE=trusted-header`），以便 API 按真实用户隔离计划。分布式部署且配置 `MYSQL_URL` 时，计划自动使用 MySQL 持久化；否则使用 `artifacts/research_plans`。

## 安全边界

- Planner 只生成白名单中的 `paper_qa` 与 `synthesize` 任务。
- `paper_id` 必须已经位于当前可见 Paper KG；不接受模型或用户临时编造的论文 ID。
- 计划执行器不能上传文件、运行 Shell、修改图谱或发起任意网络请求。
- 子任务发生错误会标记失败并停止汇总，不用未经校验的内容填补结论。

# Query Router 与 Evidence Planner

## 目的

Router 在 Context Guard 完成论文范围与指代解析后，为问题生成受限的 `QueryPlan`。它不是让 LLM 自由选择工具，而是用可审计规则选择既有检索能力。

初版支持：定义、机制、动机、实验、比较、公式、表格、图示、多跳关系、追问与通用问题。

## 实际执行方式

1. Router 输出意图、证据类型、章节偏好、是否精排/图谱/Judge、置信度及澄清需求。
2. Evidence Planner 将复杂机制、比较和多跳问题拆为有限子问题。
3. 检索节点按 `ChunkKind` 分别检索，并优先排序目标章节；随后按 Chunk ID 去重，构造 Evidence Pack。
4. 复杂问题继续走 Citation Validator 和 Semantic Judge；定义和精确来源定位问题可跳过语义 Judge 以降低成本。
5. Trace 记录路由决策、证据计划、每类型证据命中数、去重后的证据数及最终策略。

图谱查询目前只由 `use_graph` 标记策略意图；下一阶段实现 Neo4j Graph Retriever 后，该标记将触发“图谱找候选、Chunk 做最终引用”的链路。

## 评测

```bash
python scripts/evaluate_routing.py \
  --golden evals/routing_golden.v1.json \
  --output artifacts/evals/routing.v1.json
```

草案标注集为 `evals/routing_golden.v1.draft.json`，不能直接作为正式发布门禁。人工复核并固定为 `routing_golden.v1.json` 后，建议最低门槛为：

- Intent accuracy ≥ 0.85
- Evidence type Macro-F1 ≥ 0.80
- Graph routing accuracy ≥ 0.85
- Clarification accuracy ≥ 0.90

## 当前限制

- 章节偏好是候选排序信号，不是底层向量库的硬过滤条件，避免因解析章节误差导致零召回。
- 多论文追问中的“前一个方法”仍依赖 Context Guard 的实体解析；后续应将已解析的多实体上下文显式传给 Router。
- 初版 Evidence Planner 使用确定性拆分，尚未接入可选 LLM Planner；因此不会额外消耗生成模型资源。

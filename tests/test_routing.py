from paper_agent.domain import ChunkKind
from paper_agent.routing import EvidencePlanner, QueryIntent, QueryRouter


def test_router_routes_formula_to_equation_and_context_text():
    plan = QueryRouter().route("解释式(3)的训练目标", paper_id="paper-flow")

    assert plan.intent == QueryIntent.FORMULA_EXPLANATION
    assert plan.required_evidence_types == [ChunkKind.EQUATION, ChunkKind.TEXT]
    assert plan.use_rerank is True
    assert plan.use_judge is True


def test_router_requests_clarification_for_unscoped_table_reference():
    plan = QueryRouter().route("表2中哪个方法更好？")

    assert plan.needs_clarification is True
    assert plan.clarification_question is not None


def test_router_uses_lightweight_policy_for_definition():
    plan = QueryRouter().route("Q-Former 是什么？", paper_id="paper-blip2")

    assert plan.intent == QueryIntent.DEFINITION
    assert plan.use_rerank is False
    assert plan.use_judge is False
    assert plan.paper_scope == ["paper-blip2"]


def test_evidence_planner_expands_mechanism_without_unbounded_tasks():
    route = QueryRouter().route("How does Q-Former bridge vision and language?", paper_id="paper-blip2")
    plan = EvidencePlanner().plan("How does Q-Former bridge vision and language?", route)

    assert plan.intent == QueryIntent.MECHANISM
    assert len(plan.sub_questions) == 3
    assert plan.minimum_evidence_count == 3
    assert all(item.evidence_types == [ChunkKind.TEXT, ChunkKind.FIGURE] for item in plan.sub_questions)

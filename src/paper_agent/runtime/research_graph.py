"""基于 LangGraph 的证据问答与独立审查状态图。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from paper_agent.agents import CitationValidator
from paper_agent.domain import CitationValidation, EvidencePack, GroundedAnswer, SemanticCitationReport
from paper_agent.memory import ContextGuardDecision
from paper_agent.observability import TraceRecorder
from paper_agent.routing import EvidencePlan, EvidencePlanner, QueryPlan, QueryRouter


AnswerCallable = Callable[[EvidencePack, str | None], tuple[GroundedAnswer, CitationValidation]]
ContextCallable = Callable[[str, str | None], ContextGuardDecision]
RetrieveCallable = Callable[[str, int, str | None, object | None, bool], list[object]]
EvidenceCallable = Callable[[str, list[object]], EvidencePack]
JudgeCallable = Callable[[GroundedAnswer, EvidencePack], SemanticCitationReport]
ToolRunCallable = Callable[[str, Callable[[], Any]], Any]
RouteCallable = Callable[[str, str | None, object | None, bool], QueryPlan]
EvidencePlanCallable = Callable[[str, QueryPlan], EvidencePlan]


@dataclass(frozen=True)
class ResearchGraphServices:
    """状态图依赖的受控业务能力，不向模型暴露任意执行权限。"""

    resolve_context: ContextCallable
    retrieve: RetrieveCallable
    build_evidence: EvidenceCallable
    answer: AnswerCallable
    judge: JudgeCallable | None = None
    run_tool: ToolRunCallable | None = None
    route: RouteCallable | None = None
    plan_evidence: EvidencePlanCallable | None = None


class ResearchGraphRun:
    """从状态图最终状态读取 API 所需的稳定结果。"""

    def __init__(self, state: "ResearchGraphState") -> None:
        self.context = state.get("context")
        self.evidence_pack = state["evidence_pack"]
        self.answer = state["answer"]
        self.citation_validation = state["citation_validation"]
        self.semantic_report = state.get("semantic_report")
        self.attempts = state.get("attempts", 0)
        self.semantic_gate_enabled = state.get("semantic_gate_enabled", False)
        self.semantic_gate_passed = state.get("semantic_gate_passed")
        self.refusal_kind = state.get("refusal_kind")
        self.rewrite_count = state.get("rewrite_count", 0)
        self.route_plan = state.get("route_plan")
        self.evidence_plan = state.get("evidence_plan")


class ResearchGraphState(TypedDict, total=False):
    query: str
    paper_id: str | None
    kind: object | None
    top_k: int
    use_rerank: bool
    max_attempts: int
    context: ContextGuardDecision
    effective_paper_id: str | None
    route_plan: QueryPlan
    evidence_plan: EvidencePlan
    router_clarification_question: str | None
    evidence_pack: EvidencePack
    answer: GroundedAnswer
    citation_validation: CitationValidation
    semantic_report: SemanticCitationReport | None
    semantic_gate_enabled: bool
    semantic_gate_passed: bool | None
    feedback: str | None
    attempts: int
    rewrite_count: int
    refusal_kind: str | None
    trace_recorder: TraceRecorder


def build_research_graph(services: ResearchGraphServices):
    """构建“消歧—检索—回答—引用校验—语义审查—重写/拒答”的确定性状态图。"""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("请安装 LangGraph：pip install -e '.[agent]'") from exc

    def execute(name: str, callback: Callable[[], Any]) -> Any:
        """所有图节点经同一受控工具入口运行，禁止 LLM 自行指定工具。"""

        return services.run_tool(name, callback) if services.run_tool else callback()

    def resolve_context(state: ResearchGraphState) -> dict[str, Any]:
        decision = execute(
            "resolve_context",
            lambda: services.resolve_context(state["query"], state.get("paper_id")),
        )
        return {"context": decision, "effective_paper_id": decision.resolved_paper_id}

    def clarification(state: ResearchGraphState) -> dict[str, Any]:
        decision = state["context"]
        answer = GroundedAnswer(
            answer=state.get("router_clarification_question") or decision.clarification_question or "请说明你指的是哪篇论文或哪个方法。",
            abstained=True,
            abstention_reason="问题依赖缺失的对话上下文，不能安全检索。",
        )
        pack = EvidencePack(query=state["query"], items=[])
        return {
            "evidence_pack": pack,
            "answer": answer,
            "citation_validation": CitationValidator().validate(answer, pack),
            "attempts": 0,
            "rewrite_count": 0,
            "refusal_kind": "needs_clarification",
            "semantic_gate_enabled": False,
            "semantic_gate_passed": None,
        }

    def route_query(state: ResearchGraphState) -> dict[str, Any]:
        route = execute(
            "route_query",
            lambda: (
                services.route(
                    state["query"],
                    state.get("effective_paper_id"),
                    state.get("kind"),
                    state.get("use_rerank", True),
                )
                if services.route
                else QueryRouter().route(
                    state["query"],
                    paper_id=state.get("effective_paper_id"),
                    requested_kind=state.get("kind"),
                    requested_rerank=state.get("use_rerank", True),
                )
            ),
        )
        recorder = state.get("trace_recorder")
        if recorder:
            recorder.record_event(
                "route_decision",
                status="success",
                elapsed_ms=0,
                attributes={
                    "intent": route.intent.value,
                    "evidence_types": [item.value for item in route.required_evidence_types],
                    "use_rerank": route.use_rerank,
                    "use_graph": route.use_graph,
                    "use_judge": route.use_judge,
                    "confidence": route.confidence,
                    "needs_clarification": route.needs_clarification,
                },
            )
        return {
            "route_plan": route,
            "router_clarification_question": route.clarification_question,
            "effective_paper_id": route.paper_scope[0] if len(route.paper_scope) == 1 else state.get("effective_paper_id"),
        }

    def plan_evidence(state: ResearchGraphState) -> dict[str, Any]:
        plan = execute(
            "plan_evidence",
            lambda: (
                services.plan_evidence(state["query"], state["route_plan"])
                if services.plan_evidence
                else EvidencePlanner().plan(state["query"], state["route_plan"])
            ),
        )
        recorder = state.get("trace_recorder")
        if recorder:
            recorder.record_event(
                "evidence_plan",
                status="success",
                elapsed_ms=0,
                attributes={
                    "sub_question_count": len(plan.sub_questions),
                    "minimum_evidence_count": plan.minimum_evidence_count,
                    "use_graph": plan.use_graph,
                },
            )
        return {"evidence_plan": plan}

    def retrieve(state: ResearchGraphState) -> dict[str, Any]:
        plan = state["evidence_plan"]
        total_paths = sum(len(item.evidence_types) for item in plan.sub_questions)
        per_path_top_k = max(1, state["top_k"] // max(1, total_paths))
        candidate_top_k = min(5, max(3, per_path_top_k))
        merged_hits: list[object] = []
        hit_ids: set[str] = set()
        kind_counts: dict[str, int] = {}
        for sub_question in plan.sub_questions:
            for kind in sub_question.evidence_types:
                hits = execute(
                    "retrieve_evidence",
                    lambda query=sub_question.query, evidence_kind=kind: services.retrieve(
                        query,
                        candidate_top_k,
                        state.get("effective_paper_id"),
                        evidence_kind,
                        state["route_plan"].use_rerank,
                    ),
                )
                for hit in _prioritize_sections(hits, sub_question.preferred_sections):
                    chunk_id = str(getattr(hit, "chunk_id", ""))
                    if chunk_id and chunk_id not in hit_ids:
                        hit_ids.add(chunk_id)
                        merged_hits.append(hit)
                        kind_counts[str(getattr(hit, "kind", "unknown"))] = kind_counts.get(str(getattr(hit, "kind", "unknown")), 0) + 1
        hits = merged_hits[: max(state["top_k"], plan.minimum_evidence_count)]
        pack = execute("build_evidence", lambda: services.build_evidence(state["query"], hits))
        recorder = state.get("trace_recorder")
        if recorder:
            recorder.record_event(
                "evidence_merge",
                status="success",
                elapsed_ms=0,
                attributes={
                    "retrieval_paths": total_paths,
                    "candidate_top_k_per_path": candidate_top_k,
                    "deduplicated_hit_count": len(merged_hits),
                    "evidence_count": len(pack.items),
                    "kind_counts": kind_counts,
                    "minimum_evidence_count": plan.minimum_evidence_count,
                },
            )
        return {"evidence_pack": pack}

    def generate(state: ResearchGraphState) -> dict[str, Any]:
        answer, validation = execute(
            "generate_answer",
            lambda: services.answer(state["evidence_pack"], state.get("feedback")),
        )
        return {
            "answer": answer,
            "citation_validation": validation,
            "attempts": state.get("attempts", 0) + 1,
            "rewrite_count": state.get("rewrite_count", 0) + int(bool(state.get("feedback"))),
        }

    def validate_citations(state: ResearchGraphState) -> dict[str, Any]:
        validation = execute(
            "validate_citations",
            lambda: CitationValidator().validate(state["answer"], state["evidence_pack"]),
        )
        return {
            "citation_validation": validation,
            "refusal_kind": None if validation.valid else "citation_integrity_failed",
        }

    def judge(state: ResearchGraphState) -> dict[str, Any]:
        if services.judge is None or state["answer"].abstained or not state["route_plan"].use_judge:
            return {
                "semantic_report": None,
                "semantic_gate_enabled": bool(services.judge),
                "semantic_gate_passed": None,
            }
        report = execute("semantic_judge", lambda: services.judge(state["answer"], state["evidence_pack"]))
        return {
            "semantic_report": report,
            "semantic_gate_enabled": True,
            "semantic_gate_passed": report.all_supported,
            # 反馈只指出不受支持的 Claim；已通过的 Claim 不要求模型改写。
            "feedback": _judge_feedback(report),
            "refusal_kind": None if report.all_supported else "semantic_support_failed",
        }

    def refuse(state: ResearchGraphState) -> dict[str, Any]:
        refusal_kind = state.get("refusal_kind") or "evidence_insufficient"
        reasons = {
            "citation_integrity_failed": "生成结果未通过确定性引用完整性校验。",
            "semantic_support_failed": "独立语义审查在最大重写次数后仍发现不受支持的 Claim。",
            "evidence_insufficient": "检索证据不足以支持可靠回答。",
        }
        answer = GroundedAnswer(
            answer="现有证据不足以给出可追溯的可靠回答，因此我选择拒答。",
            abstained=True,
            abstention_reason=reasons.get(refusal_kind, reasons["evidence_insufficient"]),
        )
        return {
            "answer": answer,
            "citation_validation": CitationValidator().validate(answer, state["evidence_pack"]),
            "semantic_gate_passed": False,
            "refusal_kind": refusal_kind,
        }

    def after_context(state: ResearchGraphState) -> str:
        return "clarification" if state["context"].needs_clarification else "route"

    def after_route(state: ResearchGraphState) -> str:
        return "clarification" if state["route_plan"].needs_clarification else "plan_evidence"

    def after_validation(state: ResearchGraphState) -> str:
        return "refuse" if not state["citation_validation"].valid else "judge"

    def after_judge(state: ResearchGraphState) -> str:
        if not state.get("semantic_gate_enabled") or state["answer"].abstained:
            return "end"
        if state.get("semantic_gate_passed"):
            return "end"
        return "generate" if state.get("attempts", 0) < state["max_attempts"] else "refuse"

    def traced(name: str, handler):
        """为每个节点记录耗时与失败状态，不写入原始问题或论文正文。"""

        def wrapped(state: ResearchGraphState):
            recorder = state.get("trace_recorder")
            return recorder.run(name, lambda: handler(state)) if recorder else handler(state)

        return wrapped

    graph = StateGraph(ResearchGraphState)
    graph.add_node("resolve_context", traced("resolve_context", resolve_context))
    graph.add_node("clarification", traced("clarification", clarification))
    graph.add_node("route_query", traced("route_query", route_query))
    graph.add_node("plan_evidence", traced("plan_evidence", plan_evidence))
    graph.add_node("retrieve_evidence", traced("retrieve_evidence", retrieve))
    graph.add_node("generate_answer", traced("generate_answer", generate))
    graph.add_node("validate_citations", traced("validate_citations", validate_citations))
    graph.add_node("semantic_judge", traced("semantic_judge", judge))
    graph.add_node("refuse", traced("refuse", refuse))
    graph.add_edge(START, "resolve_context")
    graph.add_conditional_edges("resolve_context", after_context, {"clarification": "clarification", "route": "route_query"})
    graph.add_edge("clarification", END)
    graph.add_conditional_edges("route_query", after_route, {"clarification": "clarification", "plan_evidence": "plan_evidence"})
    graph.add_edge("plan_evidence", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_answer")
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_conditional_edges("validate_citations", after_validation, {"judge": "semantic_judge", "refuse": "refuse"})
    graph.add_conditional_edges("semantic_judge", after_judge, {"end": END, "generate": "generate_answer", "refuse": "refuse"})
    graph.add_edge("refuse", END)
    return graph.compile()


def _judge_feedback(report: SemanticCitationReport) -> str:
    """将不通过项压缩为最小重写指令，减少重写引入的新幻觉。"""

    unsupported = [
        f"Claim {item.claim_index} 判定为 {item.verdict.value}：{item.reasoning_summary}"
        for item in report.assessments
        if item.verdict.value != "supported"
    ]
    if not unsupported:
        return ""
    return "仅修改下列不受支持的 Claim；保留已支持的 Claim。若无足够证据，请拒答。\n" + "\n".join(unsupported)


def _prioritize_sections(hits: list[object], preferred_sections: list[str]) -> list[object]:
    """优先保留目标章节命中的候选；无命中时不丢弃原始检索结果。"""

    if not preferred_sections:
        return hits
    normalized = [value.lower() for value in preferred_sections]

    def rank(hit: object) -> int:
        path = " > ".join(getattr(hit, "section_path", [])).lower()
        return 0 if any(value in path for value in normalized) else 1

    return sorted(hits, key=rank)

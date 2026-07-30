"""Query Router 的离线评测模型与指标计算。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from paper_agent.domain import ChunkKind
from paper_agent.routing import QueryRouter


class RoutingGoldenCase(BaseModel):
    case_id: str
    query: str
    intent: str
    paper_scope: list[str] = Field(default_factory=list)
    expected_evidence_types: list[ChunkKind]
    expected_use_graph: bool
    expected_needs_clarification: bool
    session_context: dict[str, str] = Field(default_factory=dict)


class RoutingCaseResult(BaseModel):
    case_id: str
    expected_intent: str
    predicted_intent: str
    expected_evidence_types: list[ChunkKind]
    predicted_evidence_types: list[ChunkKind]
    expected_use_graph: bool
    predicted_use_graph: bool
    expected_needs_clarification: bool
    predicted_needs_clarification: bool
    correct: bool


class RoutingEvaluation(BaseModel):
    case_count: int
    intent_accuracy: float
    evidence_type_macro_f1: float
    graph_routing_accuracy: float
    clarification_accuracy: float
    cases: list[RoutingCaseResult]


def evaluate_routing(cases: list[RoutingGoldenCase], router: QueryRouter | None = None) -> RoutingEvaluation:
    """使用与线上一致的 Router 计算意图、证据类型、图谱与澄清四项指标。"""

    if not cases:
        raise ValueError("routing golden set must not be empty")
    classifier = router or QueryRouter()
    results: list[RoutingCaseResult] = []
    for case in cases:
        # 多轮案例使用标注的 active_paper 模拟 Context Guard 已解析出的论文范围。
        paper_id = case.session_context.get("active_paper") or (case.paper_scope[0] if len(case.paper_scope) == 1 else None)
        predicted = classifier.route(case.query, paper_id=paper_id)
        expected_types = _normalized_types(case.expected_evidence_types)
        predicted_types = _normalized_types(predicted.required_evidence_types)
        correct = (
            case.intent == predicted.intent.value
            and expected_types == predicted_types
            and case.expected_use_graph == predicted.use_graph
            and case.expected_needs_clarification == predicted.needs_clarification
        )
        results.append(
            RoutingCaseResult(
                case_id=case.case_id,
                expected_intent=case.intent,
                predicted_intent=predicted.intent.value,
                expected_evidence_types=expected_types,
                predicted_evidence_types=predicted_types,
                expected_use_graph=case.expected_use_graph,
                predicted_use_graph=predicted.use_graph,
                expected_needs_clarification=case.expected_needs_clarification,
                predicted_needs_clarification=predicted.needs_clarification,
                correct=correct,
            )
        )
    return RoutingEvaluation(
        case_count=len(results),
        intent_accuracy=_mean(item.expected_intent == item.predicted_intent for item in results),
        evidence_type_macro_f1=_evidence_type_macro_f1(results),
        graph_routing_accuracy=_mean(item.expected_use_graph == item.predicted_use_graph for item in results),
        clarification_accuracy=_mean(
            item.expected_needs_clarification == item.predicted_needs_clarification for item in results
        ),
        cases=results,
    )


def _normalized_types(values: list[ChunkKind]) -> list[ChunkKind]:
    return sorted(set(values), key=str)


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _evidence_type_macro_f1(results: list[RoutingCaseResult]) -> float:
    labels = sorted({kind for item in results for kind in item.expected_evidence_types + item.predicted_evidence_types}, key=str)
    scores: list[float] = []
    for label in labels:
        true_positive = sum(label in item.expected_evidence_types and label in item.predicted_evidence_types for item in results)
        false_positive = sum(label not in item.expected_evidence_types and label in item.predicted_evidence_types for item in results)
        false_negative = sum(label in item.expected_evidence_types and label not in item.predicted_evidence_types for item in results)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0

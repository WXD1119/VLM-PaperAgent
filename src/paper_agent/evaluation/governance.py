"""Offline baselines for scope isolation and research-router dispatches."""

from __future__ import annotations

from pydantic import BaseModel, Field

from paper_agent.orchestration import ResearchRouteRequest, ResearchRouter


class ScopeLeakCase(BaseModel):
    case_id: str
    allowed_paper_ids: set[str] = Field(min_length=1)
    retrieved_paper_ids: list[str] = Field(default_factory=list)


class ScopeLeakCaseResult(BaseModel):
    case_id: str
    leaked_paper_ids: list[str]
    safe: bool


class ScopeLeakEvaluation(BaseModel):
    case_count: int
    safe_case_rate: float
    leaked_paper_rate: float
    cases: list[ScopeLeakCaseResult]


def evaluate_scope_leaks(cases: list[ScopeLeakCase]) -> ScopeLeakEvaluation:
    if not cases:
        raise ValueError("scope leak golden set must not be empty")
    results = [
        ScopeLeakCaseResult(
            case_id=case.case_id,
            leaked_paper_ids=sorted(set(case.retrieved_paper_ids) - case.allowed_paper_ids),
            safe=not (set(case.retrieved_paper_ids) - case.allowed_paper_ids),
        )
        for case in cases
    ]
    total_retrieved = sum(len(set(case.retrieved_paper_ids)) for case in cases)
    total_leaked = sum(len(result.leaked_paper_ids) for result in results)
    return ScopeLeakEvaluation(
        case_count=len(results),
        safe_case_rate=sum(result.safe for result in results) / len(results),
        leaked_paper_rate=total_leaked / total_retrieved if total_retrieved else 0.0,
        cases=results,
    )


class RouterDispatchCase(BaseModel):
    case_id: str
    request: ResearchRouteRequest
    expected_modes: list[str] = Field(min_length=1)
    expected_confirmation: bool = False
    expected_clarification: bool = False


class RouterDispatchEvaluation(BaseModel):
    case_count: int
    exact_dispatch_rate: float
    confirmation_accuracy: float
    clarification_accuracy: float


def evaluate_router_dispatches(
    cases: list[RouterDispatchCase], *, router: ResearchRouter | None = None
) -> RouterDispatchEvaluation:
    if not cases:
        raise ValueError("router dispatch golden set must not be empty")
    router = router or ResearchRouter()
    decisions = [(case, router.route(case.request)) for case in cases]
    return RouterDispatchEvaluation(
        case_count=len(cases),
        exact_dispatch_rate=sum(
            [dispatch.mode for dispatch in decision.dispatches] == case.expected_modes
            for case, decision in decisions
        ) / len(cases),
        confirmation_accuracy=sum(
            any(dispatch.needs_confirmation for dispatch in decision.dispatches) == case.expected_confirmation
            for case, decision in decisions
        ) / len(cases),
        clarification_accuracy=sum(
            decision.needs_clarification == case.expected_clarification
            for case, decision in decisions
        ) / len(cases),
    )

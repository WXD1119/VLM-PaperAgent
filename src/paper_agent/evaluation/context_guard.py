from statistics import mean

from pydantic import BaseModel, Field

from paper_agent.memory import ContextGuard, ContextGuardAction, SessionState, UserProfile


class ContextGuardCase(BaseModel):
    case_id: str
    query: str
    expected_action: ContextGuardAction
    session_current_paper_id: str | None = None
    explicit_paper_id: str | None = None
    preferred_language: str | None = None
    expected_resolved_paper_id: str | None = None
    expected_needs_clarification: bool = False
    notes: str = ""


class ContextGuardCaseResult(BaseModel):
    case_id: str
    query: str
    expected_action: ContextGuardAction
    predicted_action: ContextGuardAction
    expected_resolved_paper_id: str | None = None
    predicted_resolved_paper_id: str | None = None
    expected_needs_clarification: bool
    predicted_needs_clarification: bool
    correct: bool


class ContextGuardEvaluation(BaseModel):
    case_count: int
    accuracy: float
    constraint_recall: float
    clarification_recall: float
    wrong_constraint_rate: float
    cases: list[ContextGuardCaseResult] = Field(default_factory=list)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_context_guard(
    cases: list[ContextGuardCase],
    *,
    guard: ContextGuard | None = None,
) -> ContextGuardEvaluation:
    if not cases:
        raise ValueError("context guard golden set must contain at least one case")
    guard = guard or ContextGuard()
    results: list[ContextGuardCaseResult] = []

    for case in cases:
        decision = guard.decide(
            case.query,
            explicit_paper_id=case.explicit_paper_id,
            session=SessionState(current_paper_id=case.session_current_paper_id),
            profile=UserProfile(preferred_language=case.preferred_language),
        )
        correct = (
            decision.action == case.expected_action
            and decision.resolved_paper_id == case.expected_resolved_paper_id
            and decision.needs_clarification == case.expected_needs_clarification
        )
        results.append(
            ContextGuardCaseResult(
                case_id=case.case_id,
                query=case.query,
                expected_action=case.expected_action,
                predicted_action=decision.action,
                expected_resolved_paper_id=case.expected_resolved_paper_id,
                predicted_resolved_paper_id=decision.resolved_paper_id,
                expected_needs_clarification=case.expected_needs_clarification,
                predicted_needs_clarification=decision.needs_clarification,
                correct=correct,
            )
        )

    expected_constraints = [
        result for result in results if result.expected_action == ContextGuardAction.CONSTRAIN
    ]
    correct_constraints = [
        result
        for result in expected_constraints
        if result.predicted_action == ContextGuardAction.CONSTRAIN
        and result.predicted_resolved_paper_id == result.expected_resolved_paper_id
    ]
    expected_clarifications = [
        result
        for result in results
        if result.expected_action == ContextGuardAction.ASK_CLARIFICATION
    ]
    correct_clarifications = [
        result
        for result in expected_clarifications
        if result.predicted_action == ContextGuardAction.ASK_CLARIFICATION
        and result.predicted_needs_clarification
    ]
    # Wrong-constraint rate measures stale-session over-binding for clear new questions.
    # Cases with explicit paper_id are excluded because a resolved paper is intended.
    clear_new_questions = [
        result
        for result in results
        if result.expected_action == ContextGuardAction.PROCEED
        and result.expected_resolved_paper_id is None
    ]
    wrong_constraints = [
        result
        for result in clear_new_questions
        if result.predicted_action == ContextGuardAction.CONSTRAIN
        or result.predicted_resolved_paper_id is not None
    ]

    return ContextGuardEvaluation(
        case_count=len(results),
        accuracy=mean(result.correct for result in results),
        constraint_recall=_safe_ratio(len(correct_constraints), len(expected_constraints)),
        clarification_recall=_safe_ratio(
            len(correct_clarifications),
            len(expected_clarifications),
        ),
        wrong_constraint_rate=_safe_ratio(len(wrong_constraints), len(clear_new_questions)),
        cases=results,
    )

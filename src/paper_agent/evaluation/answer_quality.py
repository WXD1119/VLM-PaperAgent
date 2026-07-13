from statistics import mean

from pydantic import BaseModel, Field

from paper_agent.domain import AnswerBundle, SemanticCitationReport, SupportVerdict


class AnswerQualityCase(BaseModel):
    case_id: str
    query: str
    abstained: bool
    citation_valid: bool
    claim_count: int
    semantic_assessed_claims: int = 0
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    semantic_support_rate: float = 0.0
    unsupported_claim_rate: float = 0.0
    fully_supported: bool = False


class AnswerQualityEvaluation(BaseModel):
    answer_count: int
    claim_count: int
    citation_pass_rate: float
    abstention_rate: float
    semantic_coverage_rate: float
    semantic_support_rate: float
    unsupported_claim_rate: float
    fully_supported_answer_rate: float
    cases: list[AnswerQualityCase] = Field(default_factory=list)


class AnswerQualityGateFailure(BaseModel):
    metric: str
    actual: float
    expected: str
    reason: str


class AnswerQualityGateResult(BaseModel):
    passed: bool
    failures: list[AnswerQualityGateFailure] = Field(default_factory=list)
    risky_cases: list[str] = Field(default_factory=list)


def evaluate_answer_quality(
    answers: dict[str, AnswerBundle],
    semantic_reports: dict[str, SemanticCitationReport] | None = None,
) -> AnswerQualityEvaluation:
    if not answers:
        raise ValueError("answer quality evaluation needs at least one answer")
    semantic_reports = semantic_reports or {}
    cases: list[AnswerQualityCase] = []

    for case_id, bundle in sorted(answers.items()):
        report = semantic_reports.get(case_id)
        claim_count = len(bundle.answer.claims)
        supported = partial = unsupported = assessed = 0
        if report is not None:
            assessed = len(report.assessments)
            supported = sum(
                item.verdict == SupportVerdict.SUPPORTED for item in report.assessments
            )
            partial = sum(
                item.verdict == SupportVerdict.PARTIALLY_SUPPORTED
                for item in report.assessments
            )
            unsupported = sum(
                item.verdict == SupportVerdict.UNSUPPORTED for item in report.assessments
            )
        semantic_support_rate = _safe_ratio(supported, assessed)
        unsupported_claim_rate = _safe_ratio(partial + unsupported, assessed)
        fully_supported = (
            bundle.citation_validation.valid
            and not bundle.answer.abstained
            and claim_count > 0
            and assessed == claim_count
            and supported == claim_count
        )
        cases.append(
            AnswerQualityCase(
                case_id=case_id,
                query=bundle.evidence_pack.query,
                abstained=bundle.answer.abstained,
                citation_valid=bundle.citation_validation.valid,
                claim_count=claim_count,
                semantic_assessed_claims=assessed,
                supported_claims=supported,
                partially_supported_claims=partial,
                unsupported_claims=unsupported,
                semantic_support_rate=semantic_support_rate,
                unsupported_claim_rate=unsupported_claim_rate,
                fully_supported=fully_supported,
            )
        )

    claim_count = sum(case.claim_count for case in cases)
    assessed_count = sum(case.semantic_assessed_claims for case in cases)
    supported_count = sum(case.supported_claims for case in cases)
    unsupported_count = sum(
        case.partially_supported_claims + case.unsupported_claims for case in cases
    )
    return AnswerQualityEvaluation(
        answer_count=len(cases),
        claim_count=claim_count,
        citation_pass_rate=mean(case.citation_valid for case in cases),
        abstention_rate=mean(case.abstained for case in cases),
        semantic_coverage_rate=_safe_ratio(assessed_count, claim_count),
        semantic_support_rate=_safe_ratio(supported_count, assessed_count),
        unsupported_claim_rate=_safe_ratio(unsupported_count, assessed_count),
        fully_supported_answer_rate=mean(case.fully_supported for case in cases),
        cases=cases,
    )


def evaluate_answer_quality_gate(
    evaluation: AnswerQualityEvaluation,
    *,
    min_semantic_coverage: float | None = None,
    min_semantic_support: float | None = None,
    max_unsupported_claim_rate: float | None = None,
    min_fully_supported_answer_rate: float | None = None,
) -> AnswerQualityGateResult:
    failures: list[AnswerQualityGateFailure] = []
    _check_min(
        failures,
        "semantic_coverage_rate",
        evaluation.semantic_coverage_rate,
        min_semantic_coverage,
        "not enough claims have semantic judge reports",
    )
    _check_min(
        failures,
        "semantic_support_rate",
        evaluation.semantic_support_rate,
        min_semantic_support,
        "too few assessed claims are fully supported by evidence",
    )
    _check_max(
        failures,
        "unsupported_claim_rate",
        evaluation.unsupported_claim_rate,
        max_unsupported_claim_rate,
        "too many assessed claims are unsupported or only partially supported",
    )
    _check_min(
        failures,
        "fully_supported_answer_rate",
        evaluation.fully_supported_answer_rate,
        min_fully_supported_answer_rate,
        "too few answers are citation-valid and fully semantically supported",
    )
    return AnswerQualityGateResult(
        passed=not failures,
        failures=failures,
        risky_cases=find_risky_answer_cases(evaluation),
    )


def find_risky_answer_cases(evaluation: AnswerQualityEvaluation) -> list[str]:
    risky: list[str] = []
    for case in evaluation.cases:
        missing_semantic_judgments = (
            not case.abstained
            and case.claim_count > 0
            and case.semantic_assessed_claims < case.claim_count
        )
        if (
            not case.citation_valid
            or case.unsupported_claims
            or case.partially_supported_claims
            or missing_semantic_judgments
        ):
            risky.append(case.case_id)
    return risky


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _check_min(
    failures: list[AnswerQualityGateFailure],
    metric: str,
    actual: float,
    expected: float | None,
    reason: str,
) -> None:
    if expected is not None and actual < expected:
        failures.append(
            AnswerQualityGateFailure(
                metric=metric,
                actual=actual,
                expected=f">= {expected:.4f}",
                reason=reason,
            )
        )


def _check_max(
    failures: list[AnswerQualityGateFailure],
    metric: str,
    actual: float,
    expected: float | None,
    reason: str,
) -> None:
    if expected is not None and actual > expected:
        failures.append(
            AnswerQualityGateFailure(
                metric=metric,
                actual=actual,
                expected=f"<= {expected:.4f}",
                reason=reason,
            )
        )

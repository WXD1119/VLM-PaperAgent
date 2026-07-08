from statistics import mean

from pydantic import BaseModel, Field

from paper_agent.domain.answer import SemanticCitationReport, SupportVerdict


class CitationClaimLabel(BaseModel):
    claim_index: int = Field(ge=1)
    verdict: SupportVerdict
    notes: str = ""


class CitationCaseLabel(BaseModel):
    case_id: str
    query: str
    answer_abstained: bool
    abstention_correct: bool
    claims: list[CitationClaimLabel] = Field(default_factory=list)


class CitationGoldenSet(BaseModel):
    version: str = "v1"
    annotator: str
    cases: list[CitationCaseLabel] = Field(default_factory=list)


class CitationEvaluation(BaseModel):
    case_count: int
    claim_count: int
    claim_accuracy: float
    macro_f1: float
    supported_precision: float
    supported_recall: float
    abstention_accuracy: float


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_citation_judge(
    golden: CitationGoldenSet,
    predictions: dict[str, SemanticCitationReport],
) -> CitationEvaluation:
    if not golden.cases:
        raise ValueError("citation golden set must contain at least one case")
    expected: list[SupportVerdict] = []
    predicted: list[SupportVerdict] = []
    for case in golden.cases:
        if case.case_id not in predictions:
            raise ValueError(f"missing judge prediction for case: {case.case_id}")
        report = predictions[case.case_id]
        by_index = {item.claim_index: item.verdict for item in report.assessments}
        if len(by_index) != len(report.assessments):
            raise ValueError(f"duplicate claim index in prediction: {case.case_id}")
        for label in case.claims:
            if label.claim_index not in by_index:
                raise ValueError(
                    f"missing claim {label.claim_index} prediction for case: {case.case_id}"
                )
            expected.append(label.verdict)
            predicted.append(by_index[label.claim_index])

    correct = sum(left == right for left, right in zip(expected, predicted))
    f1_values: list[float] = []
    for verdict in SupportVerdict:
        tp = sum(e == verdict and p == verdict for e, p in zip(expected, predicted))
        fp = sum(e != verdict and p == verdict for e, p in zip(expected, predicted))
        fn = sum(e == verdict and p != verdict for e, p in zip(expected, predicted))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1_values.append(_safe_ratio(2 * precision * recall, precision + recall))

    supported = SupportVerdict.SUPPORTED
    supported_tp = sum(e == supported and p == supported for e, p in zip(expected, predicted))
    supported_fp = sum(e != supported and p == supported for e, p in zip(expected, predicted))
    supported_fn = sum(e == supported and p != supported for e, p in zip(expected, predicted))
    return CitationEvaluation(
        case_count=len(golden.cases),
        claim_count=len(expected),
        claim_accuracy=_safe_ratio(correct, len(expected)),
        macro_f1=mean(f1_values),
        supported_precision=_safe_ratio(supported_tp, supported_tp + supported_fp),
        supported_recall=_safe_ratio(supported_tp, supported_tp + supported_fn),
        abstention_accuracy=mean(case.abstention_correct for case in golden.cases),
    )

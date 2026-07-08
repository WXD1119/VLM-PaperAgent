import pytest

from paper_agent.domain import ClaimSupportAssessment, SemanticCitationReport, SupportVerdict
from paper_agent.evaluation.citation import (
    CitationCaseLabel,
    CitationClaimLabel,
    CitationGoldenSet,
    evaluate_citation_judge,
)


def assessment(index: int, verdict: SupportVerdict) -> ClaimSupportAssessment:
    return ClaimSupportAssessment(
        claim_index=index,
        verdict=verdict,
        evidence_ids=["E1"],
        reasoning_summary="test",
    )


def test_citation_metrics_compare_human_and_judge_labels():
    golden = CitationGoldenSet(
        annotator="human",
        cases=[
            CitationCaseLabel(
                case_id="case",
                query="query",
                answer_abstained=False,
                abstention_correct=True,
                claims=[
                    CitationClaimLabel(claim_index=1, verdict=SupportVerdict.SUPPORTED),
                    CitationClaimLabel(
                        claim_index=2, verdict=SupportVerdict.PARTIALLY_SUPPORTED
                    ),
                    CitationClaimLabel(claim_index=3, verdict=SupportVerdict.UNSUPPORTED),
                ],
            )
        ],
    )
    prediction = SemanticCitationReport(
        assessments=[
            assessment(1, SupportVerdict.SUPPORTED),
            assessment(2, SupportVerdict.SUPPORTED),
            assessment(3, SupportVerdict.UNSUPPORTED),
        ]
    )
    result = evaluate_citation_judge(golden, {"case": prediction})
    assert result.claim_accuracy == pytest.approx(2 / 3)
    assert result.supported_precision == pytest.approx(0.5)
    assert result.supported_recall == 1.0
    assert result.abstention_accuracy == 1.0


def test_citation_metrics_require_prediction_for_every_case():
    golden = CitationGoldenSet(
        annotator="human",
        cases=[
            CitationCaseLabel(
                case_id="missing",
                query="query",
                answer_abstained=True,
                abstention_correct=True,
            )
        ],
    )
    with pytest.raises(ValueError, match="missing judge prediction"):
        evaluate_citation_judge(golden, {})

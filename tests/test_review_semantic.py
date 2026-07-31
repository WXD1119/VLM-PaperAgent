from paper_agent.agents import (
    ReflectionAction,
    ReviewClaim,
    ReviewEvidence,
    ReviewSectionDraft,
    ReviewSectionSemanticJudge,
)
from paper_agent.domain import ClaimSupportAssessment, SemanticCitationReport, SupportVerdict


class RecordingJudge:
    def __init__(self, report: SemanticCitationReport | Exception) -> None:
        self.report = report
        self.answer = None
        self.pack = None

    def evaluate(self, answer, pack):
        self.answer, self.pack = answer, pack
        if isinstance(self.report, Exception):
            raise self.report
        return self.report


def _draft(chunk_id="c1"):
    return ReviewSectionDraft(
        section_id="taxonomy",
        title="Taxonomy",
        content="Method A uses queries.",
        citation_valid=True,
        evidence_chunk_ids=[chunk_id],
        claims=[ReviewClaim(text="Method A uses queries.", evidence_chunk_ids=[chunk_id])],
    )


def _evidence():
    return [
        ReviewEvidence(
            section_id="taxonomy", paper_id="p1", summary="A uses learnable queries.",
            evidence_chunk_ids=["c1"], citation_valid=True, abstained=False,
        )
    ]


def _report(verdict=SupportVerdict.SUPPORTED):
    return SemanticCitationReport(assessments=[
        ClaimSupportAssessment(
            claim_index=1, verdict=verdict, evidence_ids=["c1"], reasoning_summary="judged"
        )
    ])


def test_section_semantic_judge_preserves_chunk_ids_and_passes_supported_claims():
    judge = RecordingJudge(_report())
    result = ReviewSectionSemanticJudge(judge).review(_draft(), _evidence())
    assert result.suggested_action == ReflectionAction.PASS
    assert judge.answer.claims[0].evidence_ids == ["c1"]
    assert judge.pack.items[0].evidence_id == "c1"
    assert result.as_reflection().affected_section_ids == []


def test_section_semantic_judge_requests_rewrite_for_partial_or_unsupported_claim():
    result = ReviewSectionSemanticJudge(RecordingJudge(_report(SupportVerdict.PARTIALLY_SUPPORTED))).review(_draft(), _evidence())
    assert result.suggested_action == ReflectionAction.REWRITE_SECTION
    assert result.unsupported_claim_indexes == [1]


def test_section_semantic_judge_rejects_unknown_chunk_without_model_call():
    judge = RecordingJudge(_report())
    result = ReviewSectionSemanticJudge(judge).review(_draft("unknown"), _evidence())
    assert result.suggested_action == ReflectionAction.REWRITE_SECTION
    assert judge.answer is None


def test_section_semantic_judge_handles_judge_failure_and_incomplete_audit_conservatively():
    failed = ReviewSectionSemanticJudge(RecordingJudge(RuntimeError("offline"))).review(_draft(), _evidence())
    incomplete = ReviewSectionSemanticJudge(RecordingJudge(SemanticCitationReport())).review(_draft(), _evidence())
    assert failed.suggested_action == ReflectionAction.RETRIEVE_MORE
    assert incomplete.suggested_action == ReflectionAction.RETRIEVE_MORE

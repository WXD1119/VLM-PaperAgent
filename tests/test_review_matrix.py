from paper_agent.agents import (
    DeterministicReviewPlanner,
    EvidenceMatrixBuilder,
    ReviewCitationCoverageValidator,
    ReviewClaim,
    ReviewEvidence,
    ReviewSectionDraft,
    ReviewTemplate,
)
from paper_agent.domain import ClaimSupportAssessment, SemanticCitationReport, SupportVerdict


def _plan():
    return DeterministicReviewPlanner().plan("retrieval", ["p1", "p2"], ReviewTemplate.RELATED_WORK)


def test_evidence_matrix_exposes_empty_cells_and_only_valid_evidence():
    plan = _plan()
    matrix = EvidenceMatrixBuilder().build(
        plan,
        [ReviewEvidence(section_id="positioning", paper_id="p1", summary="setting", evidence_chunk_ids=["c1"], citation_valid=True, abstained=False)],
    )
    assert len(matrix.cells) == sum(len(section.evidence_dimensions) for section in plan.sections) * 2
    assert matrix.allowed_chunk_ids("positioning") == {"c1"}
    assert matrix.covered_papers("positioning") == {"p1"}


def test_section_validator_rejects_unknown_claim_citation_and_reports_coverage():
    plan = _plan()
    evidence = [ReviewEvidence(section_id="positioning", paper_id="p1", summary="setting", evidence_chunk_ids=["c1"], citation_valid=True, abstained=False)]
    matrix = EvidenceMatrixBuilder().build(plan, evidence)
    draft = ReviewSectionDraft(
        section_id="positioning", title="Research positioning", content="Claim.", citation_valid=True,
        evidence_chunk_ids=["c1", "unknown"], claims=[ReviewClaim(text="Claim.", evidence_chunk_ids=["unknown"])],
    )
    result = ReviewCitationCoverageValidator().validate(plan, matrix, [draft], evidence)
    positioning = result.sections[0]
    assert positioning.unknown_evidence_chunk_ids == ["unknown"]
    assert positioning.corpus_coverage == 0
    assert result.valid is False
    assert result.affected_section_ids == ["positioning", "taxonomy", "gaps"]


def test_section_validator_uses_semantic_judge_after_allow_list_check():
    plan = _plan()
    evidence = [ReviewEvidence(section_id="positioning", paper_id="p1", summary="setting", evidence_chunk_ids=["c1"], citation_valid=True, abstained=False)]
    matrix = EvidenceMatrixBuilder().build(plan, evidence)
    draft = ReviewSectionDraft(section_id="positioning", title="Research positioning", content="Claim.", citation_valid=True, evidence_chunk_ids=["c1"], claims=[ReviewClaim(text="Claim.", evidence_chunk_ids=["c1"])])

    def judge(_draft, _evidence):
        return SemanticCitationReport(assessments=[ClaimSupportAssessment(claim_index=1, verdict=SupportVerdict.PARTIALLY_SUPPORTED, evidence_ids=["c1"], reasoning_summary="too broad")])

    result = ReviewCitationCoverageValidator(judge).validate(plan, matrix, [draft], evidence)
    section = result.sections[0]
    assert section.unsupported_claim_indexes == [1]
    assert section.supported_claim_count == 0
    assert "semantic judge found unsupported or partial claims" in section.errors

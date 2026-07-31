"""Semantic citation audit and bounded repair advice for review sections.

The review writer uses chunk IDs directly, while ``SemanticCitationJudge`` uses
short evidence IDs.  This adapter deliberately preserves the chunk ID as the
evidence ID so a judge assessment can be traced back to the review artifact
without a second ID translation layer.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from paper_agent.agents.review_writer import (
    ReflectionAction,
    ReviewClaim,
    ReviewEvidence,
    ReviewReflection,
    ReviewSectionDraft,
)
from paper_agent.domain import (
    AnswerClaim,
    ChunkKind,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)


class CitationJudgeProtocol(Protocol):
    """The subset of :class:`SemanticCitationJudge` used by review auditing."""

    def evaluate(self, answer: GroundedAnswer, pack: EvidencePack) -> SemanticCitationReport: ...


class SectionSemanticReview(BaseModel):
    """Auditable semantic result plus the next bounded workflow action."""

    section_id: str
    report: SemanticCitationReport = Field(default_factory=SemanticCitationReport)
    suggested_action: ReflectionAction
    reasoning: str
    unsupported_claim_indexes: list[int] = Field(default_factory=list)

    def as_reflection(self) -> ReviewReflection:
        return ReviewReflection(
            action=self.suggested_action,
            affected_section_ids=[] if self.suggested_action == ReflectionAction.PASS else [self.section_id],
            reasoning=self.reasoning,
        )


class ReviewSectionSemanticJudge:
    """Adapt the single-paper semantic judge to one evidence-bound review section.

    It never lets a semantic model invent a chunk identifier: the deterministic
    allow-list is checked before calling the model.  A judge failure is treated
    as a retrievable audit failure, not as a successful semantic validation.
    """

    def __init__(self, judge: CitationJudgeProtocol) -> None:
        self.judge = judge

    def evaluate(
        self,
        draft: ReviewSectionDraft,
        evidence: list[ReviewEvidence],
    ) -> SemanticCitationReport:
        """Return the semantic report for compatibility with coverage validation."""
        return self.review(draft, evidence).report

    def review(
        self,
        draft: ReviewSectionDraft,
        evidence: list[ReviewEvidence],
    ) -> SectionSemanticReview:
        allowed, pack = self._build_evidence_pack(draft, evidence)
        if not draft.claims:
            return SectionSemanticReview(
                section_id=draft.section_id,
                suggested_action=ReflectionAction.REWRITE_SECTION,
                reasoning="citation-valid review prose requires structured claims for semantic audit",
            )

        claim_ids = {
            chunk_id for claim in draft.claims for chunk_id in claim.evidence_chunk_ids
        }
        unknown = sorted(claim_ids - allowed)
        if unknown:
            return SectionSemanticReview(
                section_id=draft.section_id,
                suggested_action=ReflectionAction.REWRITE_SECTION,
                reasoning=(
                    "section claims cite chunk IDs outside the evidence allow-list: "
                    + ", ".join(unknown)
                ),
                unsupported_claim_indexes=[
                    index
                    for index, claim in enumerate(draft.claims, start=1)
                    if not set(claim.evidence_chunk_ids).issubset(allowed)
                ],
            )
        if not pack.items:
            return SectionSemanticReview(
                section_id=draft.section_id,
                suggested_action=ReflectionAction.MARK_INSUFFICIENT_EVIDENCE,
                reasoning="the section has no citation-validated evidence to audit",
                unsupported_claim_indexes=list(range(1, len(draft.claims) + 1)),
            )

        answer = GroundedAnswer(
            answer=draft.content or draft.title,
            claims=[self._to_answer_claim(claim) for claim in draft.claims],
        )
        try:
            report = self.judge.evaluate(answer, pack)
        except Exception as exc:
            return SectionSemanticReview(
                section_id=draft.section_id,
                suggested_action=ReflectionAction.RETRIEVE_MORE,
                reasoning=f"semantic citation audit was unavailable: {type(exc).__name__}",
                unsupported_claim_indexes=list(range(1, len(draft.claims) + 1)),
            )

        assessed = {item.claim_index for item in report.assessments}
        supported = {
            item.claim_index
            for item in report.assessments
            if item.verdict == SupportVerdict.SUPPORTED
        }
        expected = set(range(1, len(draft.claims) + 1))
        unsupported = sorted(expected - supported)
        if assessed != expected:
            return SectionSemanticReview(
                section_id=draft.section_id,
                report=report,
                suggested_action=ReflectionAction.RETRIEVE_MORE,
                reasoning="semantic citation audit did not assess every structured claim",
                unsupported_claim_indexes=unsupported,
            )
        if unsupported:
            return SectionSemanticReview(
                section_id=draft.section_id,
                report=report,
                suggested_action=ReflectionAction.REWRITE_SECTION,
                reasoning="semantic citation audit found unsupported or overstated claims",
                unsupported_claim_indexes=unsupported,
            )
        return SectionSemanticReview(
            section_id=draft.section_id,
            report=report,
            suggested_action=ReflectionAction.PASS,
            reasoning="every structured claim is semantically supported by its cited evidence",
        )

    @staticmethod
    def _to_answer_claim(claim: ReviewClaim) -> AnswerClaim:
        return AnswerClaim(text=claim.text, evidence_ids=claim.evidence_chunk_ids)

    @staticmethod
    def _build_evidence_pack(
        draft: ReviewSectionDraft,
        evidence: list[ReviewEvidence],
    ) -> tuple[set[str], EvidencePack]:
        items: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in evidence:
            if (
                item.section_id != draft.section_id
                or not item.citation_valid
                or item.abstained
            ):
                continue
            for chunk_id in item.evidence_chunk_ids:
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                items.append(
                    EvidenceItem(
                        evidence_id=chunk_id,
                        chunk_id=chunk_id,
                        paper_id=item.paper_id,
                        kind=ChunkKind.TEXT,
                        pages=[1],
                        section_path=["Review evidence", draft.section_id],
                        content=item.summary,
                    )
                )
        return seen, EvidencePack(query=f"Semantic audit for {draft.title}", items=items)

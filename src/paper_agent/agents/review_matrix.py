"""Evidence-matrix and citation/coverage checks for literature-review sections.

This module is deliberately independent from the LangGraph review workflow.  The
workflow can build a matrix after retrieval, validate drafts after writing, and
use :meth:`ReviewValidationResult.to_reflection_input` to decide a bounded
repair action without giving a writer permission to invent citations.
"""

from __future__ import annotations

from collections import defaultdict

from collections.abc import Callable

from pydantic import BaseModel, Field

from paper_agent.agents.review_writer import (
    LiteratureReviewPlan,
    ReviewEvidence,
    ReviewSectionDraft,
)
from paper_agent.domain import SemanticCitationReport, SupportVerdict


class EvidenceMatrixCell(BaseModel):
    """One bounded evidence cell for a section, dimension, and source paper."""

    section_id: str
    dimension: str
    paper_id: str
    summaries: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_valid: bool = False
    abstained: bool = False


class ReviewEvidenceMatrix(BaseModel):
    """Inspectable retrieval output, before any prose is generated."""

    topic: str
    paper_ids: list[str]
    cells: list[EvidenceMatrixCell] = Field(default_factory=list)

    def allowed_chunk_ids(self, section_id: str) -> set[str]:
        return {
            chunk_id
            for cell in self.cells
            if cell.section_id == section_id and cell.citation_valid and not cell.abstained
            for chunk_id in cell.evidence_chunk_ids
        }

    def covered_papers(self, section_id: str) -> set[str]:
        return {
            cell.paper_id
            for cell in self.cells
            if cell.section_id == section_id and cell.citation_valid and not cell.abstained and cell.evidence_chunk_ids
        }


class EvidenceMatrixBuilder:
    """Materialize retrieval results into an auditable section/dimension matrix.

    ``ReviewEvidence`` is intentionally section-scoped.  A retrieval task may
    answer more than one requested dimension, so its result is copied into each
    requested dimension rather than guessed/classified post hoc.  Consumers can
    therefore see exactly what was available to the section writer.
    """

    def build(self, plan: LiteratureReviewPlan, evidence: list[ReviewEvidence]) -> ReviewEvidenceMatrix:
        sections = {section.section_id: section for section in plan.sections}
        grouped: dict[tuple[str, str, str], list[ReviewEvidence]] = defaultdict(list)
        for item in evidence:
            section = sections.get(item.section_id)
            if not section or item.paper_id not in plan.paper_ids:
                continue
            for dimension in section.evidence_dimensions:
                grouped[(item.section_id, dimension, item.paper_id)].append(item)

        cells: list[EvidenceMatrixCell] = []
        # Include empty cells: this makes missing coverage visible in the UI and
        # lets reflection request only the exact missing section/dimension pair.
        for section in plan.sections:
            for dimension in section.evidence_dimensions:
                for paper_id in plan.paper_ids:
                    items = grouped.get((section.section_id, dimension, paper_id), [])
                    ids = list(dict.fromkeys(chunk_id for item in items for chunk_id in item.evidence_chunk_ids))
                    usable = [item for item in items if item.citation_valid and not item.abstained and item.evidence_chunk_ids]
                    cells.append(
                        EvidenceMatrixCell(
                            section_id=section.section_id,
                            dimension=dimension,
                            paper_id=paper_id,
                            summaries=[item.summary for item in usable],
                            evidence_chunk_ids=ids if usable else [],
                            citation_valid=bool(usable),
                            abstained=bool(items) and not bool(usable) and all(item.abstained for item in items),
                        )
                    )
        return ReviewEvidenceMatrix(topic=plan.topic, paper_ids=plan.paper_ids, cells=cells)


class SectionCitationCoverage(BaseModel):
    section_id: str
    claim_count: int = Field(ge=0)
    cited_claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    citation_coverage: float = Field(ge=0, le=1)
    corpus_coverage: float = Field(ge=0, le=1)
    cited_paper_ids: list[str] = Field(default_factory=list)
    unknown_evidence_chunk_ids: list[str] = Field(default_factory=list)
    unsupported_claim_indexes: list[int] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors and not self.unknown_evidence_chunk_ids and not self.unsupported_claim_indexes


class ReviewValidationResult(BaseModel):
    sections: list[SectionCitationCoverage] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(section.valid for section in self.sections)

    @property
    def affected_section_ids(self) -> list[str]:
        return [section.section_id for section in self.sections if not section.valid]

    def to_reflection_input(self) -> dict[str, object]:
        """Small stable handoff contract for the review reflection node."""
        return {
            "valid": self.valid,
            "affected_section_ids": self.affected_section_ids,
            "sections": [section.model_dump(mode="json") for section in self.sections],
        }


SectionSemanticJudge = Callable[[ReviewSectionDraft, list[ReviewEvidence]], SemanticCitationReport]


class ReviewCitationCoverageValidator:
    """Validate per-claim citation allow-lists and section corpus coverage.

    A supplied semantic judge is optional and only runs after deterministic
    allow-list validation.  Its assessments must be one-based claim indexes,
    matching :class:`SemanticCitationReport` used by the single-paper QA path.
    """

    def __init__(self, semantic_judge: SectionSemanticJudge | None = None) -> None:
        self.semantic_judge = semantic_judge

    def validate(
        self,
        plan: LiteratureReviewPlan,
        matrix: ReviewEvidenceMatrix,
        drafts: list[ReviewSectionDraft],
        evidence: list[ReviewEvidence] | None = None,
    ) -> ReviewValidationResult:
        by_draft = {draft.section_id: draft for draft in drafts}
        by_evidence: dict[str, list[ReviewEvidence]] = defaultdict(list)
        for item in evidence or []:
            by_evidence[item.section_id].append(item)
        results = [self._validate_section(section, matrix, by_draft.get(section.section_id), by_evidence[section.section_id]) for section in plan.sections]
        return ReviewValidationResult(sections=results)

    def _validate_section(self, section, matrix: ReviewEvidenceMatrix, draft: ReviewSectionDraft | None, evidence: list[ReviewEvidence]) -> SectionCitationCoverage:
        allowed = matrix.allowed_chunk_ids(section.section_id)
        missing_dimensions = [
            dimension for dimension in section.evidence_dimensions
            if not any(cell.section_id == section.section_id and cell.dimension == dimension and cell.citation_valid for cell in matrix.cells)
        ]
        errors: list[str] = []
        if draft is None:
            return SectionCitationCoverage(
                section_id=section.section_id, claim_count=0, cited_claim_count=0, supported_claim_count=0,
                citation_coverage=0, corpus_coverage=0, missing_dimensions=missing_dimensions,
                errors=["section draft is missing"],
            )
        if draft.section_id != section.section_id:
            errors.append("draft section_id does not match the review plan")
        if draft.citation_valid is False:
            errors.append("draft is marked citation_invalid")

        unknown = sorted(set(draft.evidence_chunk_ids) - allowed)
        cited_claims = 0
        unsupported: list[int] = []
        claim_ids: set[str] = set()
        for index, claim in enumerate(draft.claims, start=1):
            ids = set(claim.evidence_chunk_ids)
            claim_ids.update(ids)
            if ids:
                cited_claims += 1
            if not ids.issubset(allowed):
                unknown.extend(ids - allowed)
                unsupported.append(index)
        unknown = sorted(set(unknown))
        if draft.claims and cited_claims != len(draft.claims):
            errors.append("one or more structured claims have no citations")
        if not draft.claims and draft.content.strip() and draft.citation_valid:
            errors.append("citation-valid prose requires structured claims for coverage validation")
        if unknown:
            errors.append("draft cites evidence outside this section allow-list")

        semantic_supported = len(draft.claims) - len(unsupported)
        if self.semantic_judge and draft.claims and not unknown:
            report = self.semantic_judge(draft, evidence)
            supported_indexes = {item.claim_index for item in report.assessments if item.verdict == SupportVerdict.SUPPORTED}
            judged_indexes = {item.claim_index for item in report.assessments}
            # Incomplete judge output is treated conservatively.
            unsupported.extend(index for index in range(1, len(draft.claims) + 1) if index not in supported_indexes)
            if len(judged_indexes) != len(draft.claims):
                errors.append("semantic judge did not assess every claim")
            if unsupported:
                errors.append("semantic judge found unsupported or partial claims")
            unsupported = sorted(set(unsupported))
            semantic_supported = len(supported_indexes)

        cited_papers = sorted({cell.paper_id for cell in matrix.cells if cell.section_id == section.section_id and set(cell.evidence_chunk_ids) & claim_ids})
        claim_count = len(draft.claims)
        return SectionCitationCoverage(
            section_id=section.section_id,
            claim_count=claim_count,
            cited_claim_count=cited_claims,
            supported_claim_count=max(0, semantic_supported),
            citation_coverage=(cited_claims / claim_count) if claim_count else 0,
            corpus_coverage=(len(cited_papers) / len(matrix.paper_ids)) if matrix.paper_ids else 0,
            cited_paper_ids=cited_papers,
            unknown_evidence_chunk_ids=unknown,
            unsupported_claim_indexes=sorted(set(unsupported)),
            missing_dimensions=missing_dimensions,
            errors=errors,
        )

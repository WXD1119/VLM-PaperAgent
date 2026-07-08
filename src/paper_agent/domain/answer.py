from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from paper_agent.domain.chunk import ChunkKind


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    paper_id: str
    kind: ChunkKind
    pages: list[int] = Field(min_length=1)
    section_path: list[str] = Field(default_factory=list)
    content: str = Field(min_length=1)
    image_path: str | None = None


class EvidencePack(BaseModel):
    query: str = Field(min_length=1)
    items: list[EvidenceItem] = Field(min_length=1)


class AnswerClaim(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    answer: str
    claims: list[AnswerClaim] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_answer_state(self) -> "GroundedAnswer":
        if self.abstained and not self.abstention_reason:
            raise ValueError("abstained answer must include abstention_reason")
        if not self.abstained and not self.claims:
            raise ValueError("non-abstained answer must include at least one claim")
        return self


class CitationValidation(BaseModel):
    valid: bool
    claim_count: int
    cited_claim_count: int
    errors: list[str] = Field(default_factory=list)


class SupportVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class ClaimSupportAssessment(BaseModel):
    claim_index: int = Field(ge=1)
    verdict: SupportVerdict
    evidence_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str = Field(min_length=1)


class SemanticCitationReport(BaseModel):
    assessments: list[ClaimSupportAssessment] = Field(default_factory=list)

    @property
    def all_supported(self) -> bool:
        return all(item.verdict == SupportVerdict.SUPPORTED for item in self.assessments)


class AnswerBundle(BaseModel):
    evidence_pack: EvidencePack
    answer: GroundedAnswer
    citation_validation: CitationValidation
    generator_model: str

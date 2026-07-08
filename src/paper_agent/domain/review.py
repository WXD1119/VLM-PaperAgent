from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Verdict(StrEnum):
    SUPPORTED = "supported"
    QUESTIONABLE = "questionable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Claim(BaseModel):
    claim_id: str
    paper_id: str
    text: str
    source_element_ids: list[str] = Field(min_length=1)


class ReviewFinding(BaseModel):
    claim_id: str
    verdict: Verdict
    evidence_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str
    confidence: float = Field(ge=0, le=1)
    review_required: bool = False

    @model_validator(mode="after")
    def require_evidence_for_supported(self) -> "ReviewFinding":
        if self.verdict == Verdict.SUPPORTED and not self.evidence_ids:
            raise ValueError("supported finding must contain evidence_ids")
        return self


class ReviewReport(BaseModel):
    paper_id: str
    findings: list[ReviewFinding] = Field(default_factory=list)


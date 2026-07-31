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
    # An empty pack is a meaningful terminal state: retrieval may find no
    # in-scope evidence, in which case the runtime returns an abstention.
    # Requiring an item here made that safe refusal path impossible to model.
    items: list[EvidenceItem] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


# 表示 有证据支撑的答案 类
class GroundedAnswer(BaseModel):
    answer: str # 完整答案内容
    claims: list[AnswerClaim] = Field(default_factory=list) # 答案中的各个主张及其证据id
    abstained: bool = False # 表示答案是否被拒绝
    abstention_reason: str | None = None # 拒绝答案的原因

    @model_validator(mode="after") # 模型校验器装饰器：在整个 GroundedAnswer 对象的字段都完成基本校验之后，再执行下面的 validate_answer_state 方法。
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

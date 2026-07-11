from enum import StrEnum

from pydantic import BaseModel, Field

from paper_agent.domain import AnswerBundle, SemanticCitationReport


class PromotionVerdict(StrEnum):
    REJECT = "reject"
    ASK_USER = "ask_user"
    ALREADY_ARCHIVED = "already_archived"


class PromotionDecision(BaseModel):
    verdict: PromotionVerdict
    reasons: list[str] = Field(default_factory=list)


class PromotionPolicy:
    """Policy for whether an answer should be offered for long-term memory.

    This policy never writes to the paper knowledge graph. It only decides whether the
    agent should ask the user before archiving or otherwise persisting an answer.
    """

    def decide(
        self,
        answer: AnswerBundle,
        semantic_report: SemanticCitationReport | None = None,
        *,
        already_archived: bool = False,
    ) -> PromotionDecision:
        reasons = []
        if already_archived:
            return PromotionDecision(
                verdict=PromotionVerdict.ALREADY_ARCHIVED,
                reasons=["answer artifact is already archived or promoted"],
            )
        if answer.answer.abstained:
            reasons.append("answer abstained")
        if not answer.answer.claims:
            reasons.append("answer has no reusable claims")
        if semantic_report is not None and not semantic_report.all_supported:
            reasons.append("semantic judge did not support all claims")
        if reasons:
            return PromotionDecision(verdict=PromotionVerdict.REJECT, reasons=reasons)
        return PromotionDecision(
            verdict=PromotionVerdict.ASK_USER,
            reasons=["answer has claims and no blocking quality issue; user confirmation required"],
        )

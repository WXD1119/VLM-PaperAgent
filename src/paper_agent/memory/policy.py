from enum import StrEnum

from pydantic import BaseModel, Field

from paper_agent.domain import AnswerBundle, SemanticCitationReport


class MemoryCandidateKind(StrEnum):
    TASK_STATE = "task_state"
    PROJECT_EVENT = "project_event"
    USER_PREFERENCE = "user_preference"
    ANSWER_ARTIFACT = "answer_artifact"
    PAPER_CONTENT = "paper_content"
    CASUAL_CHAT = "casual_chat"
    UNKNOWN = "unknown"


class MemoryAction(StrEnum):
    WRITE = "write"
    KEEP = "keep"
    ASK_USER = "ask_user"
    IGNORE = "ignore"
    ROUTE_OUTSIDE_MEMORY = "route_outside_memory"


class MemoryTarget(StrEnum):
    SESSION = "session"
    EPISODIC = "episodic"
    PROFILE = "profile"
    ARTIFACT = "artifact"
    PAPER_KG = "paper_kg"
    NONE = "none"


class MemoryCandidate(BaseModel):
    kind: MemoryCandidateKind
    content: str
    explicit_user_request: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class MemoryDecision(BaseModel):
    action: MemoryAction
    target: MemoryTarget
    reasons: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool = False


class MemoryPolicy:
    """Route candidate information to the right memory layer.

    This policy separates agent memory from the paper knowledge graph. It explains where
    information belongs, but it does not write anything by itself.
    """

    def decide(self, candidate: MemoryCandidate) -> MemoryDecision:
        if candidate.kind == MemoryCandidateKind.TASK_STATE:
            return MemoryDecision(
                action=MemoryAction.WRITE,
                target=MemoryTarget.SESSION,
                reasons=["task state is short-lived working context"],
            )
        if candidate.kind == MemoryCandidateKind.PROJECT_EVENT:
            return MemoryDecision(
                action=MemoryAction.WRITE,
                target=MemoryTarget.EPISODIC,
                reasons=["project events should be append-only episodic memory"],
            )
        if candidate.kind == MemoryCandidateKind.USER_PREFERENCE:
            if candidate.explicit_user_request:
                return MemoryDecision(
                    action=MemoryAction.WRITE,
                    target=MemoryTarget.PROFILE,
                    reasons=["explicit user preference belongs in long-term profile memory"],
                )
            return MemoryDecision(
                action=MemoryAction.ASK_USER,
                target=MemoryTarget.PROFILE,
                reasons=["inferred preference needs user confirmation before persistence"],
                requires_user_confirmation=True,
            )
        if candidate.kind == MemoryCandidateKind.ANSWER_ARTIFACT:
            return MemoryDecision(
                action=MemoryAction.KEEP,
                target=MemoryTarget.ARTIFACT,
                reasons=["generated answers stay as reproducible artifacts by default"],
            )
        if candidate.kind == MemoryCandidateKind.PAPER_CONTENT:
            return MemoryDecision(
                action=MemoryAction.ROUTE_OUTSIDE_MEMORY,
                target=MemoryTarget.PAPER_KG,
                reasons=["paper content belongs in the Paper KG, not agent memory"],
            )
        if candidate.kind == MemoryCandidateKind.CASUAL_CHAT:
            return MemoryDecision(
                action=MemoryAction.IGNORE,
                target=MemoryTarget.NONE,
                reasons=["casual chat should not pollute persistent memory"],
            )
        return MemoryDecision(
            action=MemoryAction.ASK_USER,
            target=MemoryTarget.NONE,
            reasons=["unknown memory candidate needs clarification"],
            requires_user_confirmation=True,
        )


class PromotionVerdict(StrEnum):
    REJECT = "reject"
    ASK_USER = "ask_user"
    ALREADY_ARCHIVED = "already_archived"


class PromotionDecision(BaseModel):
    verdict: PromotionVerdict
    reasons: list[str] = Field(default_factory=list)


class MemorySummary(BaseModel):
    """User-facing memory status for answer generation results.

    This is intentionally less detailed than PromotionDecision. It is safe to expose in
    CLI output or API responses without leaking developer policy internals.
    """

    status: str
    artifact_status: str = "saved"
    paper_kg_written: bool = False
    recommendation: str
    requires_user_confirmation: bool = False


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


def build_memory_summary(
    answer: AnswerBundle,
    semantic_report: SemanticCitationReport | None = None,
    *,
    already_archived: bool = False,
) -> MemorySummary:
    """Build a UI/API-safe memory summary from the answer promotion policy."""

    decision = PromotionPolicy().decide(
        answer,
        semantic_report,
        already_archived=already_archived,
    )
    if decision.verdict == PromotionVerdict.ASK_USER:
        return MemorySummary(
            status="answer saved as artifact; not written to Paper KG",
            recommendation="ask user before long-term archiving",
            requires_user_confirmation=True,
        )
    if decision.verdict == PromotionVerdict.ALREADY_ARCHIVED:
        return MemorySummary(
            status="answer saved as artifact; not written to Paper KG",
            recommendation="already archived",
        )
    return MemorySummary(
        status="answer saved as artifact; not written to Paper KG",
        recommendation="keep as artifact only; do not archive",
    )

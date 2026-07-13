from enum import StrEnum

from pydantic import BaseModel, Field

from .profile import UserProfile
from .session import SessionState


class ContextGuardAction(StrEnum):
    PROCEED = "proceed"
    CONSTRAIN = "constrain"
    ASK_CLARIFICATION = "ask_clarification"


class ContextGuardDecision(BaseModel):
    action: ContextGuardAction
    query: str
    resolved_paper_id: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    warnings: list[str] = Field(default_factory=list)
    context_notes: list[str] = Field(default_factory=list)


class ContextGuard:
    """Use agent memory to keep ambiguous follow-up questions grounded.

    The guard does not rewrite paper content and does not touch the Paper KG. It only
    uses session/profile memory to decide whether a user question should be constrained
    to the current paper or clarified before retrieval.
    """

    AMBIGUOUS_MARKERS = {
        "it",
        "this",
        "that",
        "these",
        "those",
        "above",
        "previous",
        "method",
        "model",
        "approach",
        "它",
        "这个",
        "那个",
        "这些",
        "上述",
        "上面",
        "前面",
        "这篇",
        "该方法",
        "这个方法",
        "这个模型",
        "继续",
    }

    def decide(
        self,
        query: str,
        *,
        explicit_paper_id: str | None = None,
        session: SessionState | None = None,
        profile: UserProfile | None = None,
    ) -> ContextGuardDecision:
        session = session or SessionState()
        profile = profile or UserProfile()
        normalized_query = " ".join(query.split())
        notes: list[str] = []
        warnings: list[str] = []
        if profile.preferred_language:
            notes.append(f"preferred_language={profile.preferred_language}")
        if session.current_workspace_id:
            notes.append(f"current_workspace_id={session.current_workspace_id}")

        if explicit_paper_id:
            notes.append("explicit paper_id provided by user")
            return ContextGuardDecision(
                action=ContextGuardAction.PROCEED,
                query=normalized_query,
                resolved_paper_id=explicit_paper_id,
                context_notes=notes,
            )

        ambiguous = self._is_ambiguous_followup(normalized_query)
        if ambiguous and session.current_paper_id:
            warnings.append(
                "ambiguous follow-up constrained to current_paper_id from session memory"
            )
            return ContextGuardDecision(
                action=ContextGuardAction.CONSTRAIN,
                query=normalized_query,
                resolved_paper_id=session.current_paper_id,
                warnings=warnings,
                context_notes=notes,
            )

        if ambiguous:
            return ContextGuardDecision(
                action=ContextGuardAction.ASK_CLARIFICATION,
                query=normalized_query,
                needs_clarification=True,
                clarification_question=(
                    "Which paper or concept should I use as the context for this follow-up?"
                ),
                warnings=["query appears to depend on missing conversation context"],
                context_notes=notes,
            )

        return ContextGuardDecision(
            action=ContextGuardAction.PROCEED,
            query=normalized_query,
            resolved_paper_id=None,
            context_notes=notes,
        )

    def _is_ambiguous_followup(self, query: str) -> bool:
        lowered = query.lower()
        tokens = lowered.replace("?", " ").replace("？", " ").split()
        if 0 < len(tokens) <= 3:
            return True
        return any(marker in lowered for marker in self.AMBIGUOUS_MARKERS)

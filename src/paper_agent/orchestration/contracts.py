"""Typed contracts shared between the router and research subgraphs.

The contracts deliberately carry only IDs and compact state.  Paper text remains
in the evidence store and is selected independently for each graph node.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class CorpusScope(StrEnum):
    """The only corpus boundaries accepted by the research runtime."""

    AUTO = "auto"
    TEMPORARY_WORKSPACE = "temporary_workspace"
    ACTIVE_PAPER_ONLY = "active_paper_only"
    SELECTED_PAPERS = "selected_papers"
    LIBRARY_ONLY = "library_only"
    WEB_EXPANSION = "web_expansion"


class AgentName(StrEnum):
    LIBRARY = "library"
    READING_COMPARE = "reading_compare"
    REVIEW_WRITER = "review_writer"


class ContextBudget(BaseModel):
    """A node-level budget.  It is an admission limit, not a model capacity claim."""

    total_tokens: int = Field(default=24_000, ge=2_000)
    system_tokens: int = Field(default=2_000, ge=0)
    state_tokens: int = Field(default=2_000, ge=0)
    memory_tokens: int = Field(default=2_000, ge=0)
    evidence_tokens: int = Field(default=12_000, ge=0)
    draft_tokens: int = Field(default=4_000, ge=0)
    output_reserve_tokens: int = Field(default=2_000, ge=256)

    @model_validator(mode="after")
    def validate_total(self) -> "ContextBudget":
        allocated = (
            self.system_tokens
            + self.state_tokens
            + self.memory_tokens
            + self.evidence_tokens
            + self.draft_tokens
            + self.output_reserve_tokens
        )
        if allocated > self.total_tokens:
            raise ValueError("context budget allocations exceed total_tokens")
        return self


class ResearchSession(BaseModel):
    """Compact durable state for a research workspace; never stores paper text."""

    session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    active_paper_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list)
    corpus_scope: CorpusScope = CorpusScope.AUTO
    user_goal: str | None = Field(default=None, max_length=2_000)
    unresolved_questions: list[str] = Field(default_factory=list)


class AgentDispatch(BaseModel):
    agent: AgentName
    mode: str = Field(min_length=1, max_length=128)
    corpus_scope: CorpusScope
    depends_on: list[str] = Field(default_factory=list)
    needs_confirmation: bool = False


class HandoffPacket(BaseModel):
    """Cross-agent payload containing references, not raw documents or chat logs."""

    from_agent: AgentName
    to_agent: AgentName
    session_id: str = Field(min_length=1, max_length=128)
    corpus_scope: CorpusScope
    paper_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    user_goal: str = Field(min_length=1, max_length=2_000)
    permissions_granted: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool = False


class ScopeResolution(BaseModel):
    """Resolved allow-list applied to every evidence item after retrieval."""

    scope: CorpusScope
    allowed_paper_ids: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool = False

    def allows(self, paper_id: str) -> bool:
        return not self.allowed_paper_ids or paper_id in self.allowed_paper_ids


def resolve_scope(
    scope: CorpusScope,
    *,
    active_paper_id: str | None = None,
    selected_paper_ids: list[str] | None = None,
    web_expansion_confirmed: bool = False,
) -> ScopeResolution:
    """Resolve the scope into an explicit paper allow-list.

    ``AUTO`` preserves the existing API behavior: an explicit/resolved paper is
    scoped, otherwise the request searches the library.  New workspace callers
    should always choose a non-auto scope.
    """

    selected = list(dict.fromkeys(item for item in (selected_paper_ids or []) if item))
    if scope == CorpusScope.AUTO:
        if active_paper_id:
            return ScopeResolution(scope=CorpusScope.ACTIVE_PAPER_ONLY, allowed_paper_ids=[active_paper_id])
        return ScopeResolution(scope=CorpusScope.LIBRARY_ONLY)
    if scope in {CorpusScope.TEMPORARY_WORKSPACE, CorpusScope.ACTIVE_PAPER_ONLY}:
        if not active_paper_id:
            raise ValueError(f"{scope.value} requires an active_paper_id")
        return ScopeResolution(scope=scope, allowed_paper_ids=[active_paper_id])
    if scope == CorpusScope.SELECTED_PAPERS:
        if not selected:
            raise ValueError("selected_papers requires at least one paper_id")
        return ScopeResolution(scope=scope, allowed_paper_ids=selected)
    if scope == CorpusScope.WEB_EXPANSION:
        if not web_expansion_confirmed:
            return ScopeResolution(scope=scope, requires_user_confirmation=True)
        return ScopeResolution(scope=scope)
    return ScopeResolution(scope=CorpusScope.LIBRARY_ONLY)

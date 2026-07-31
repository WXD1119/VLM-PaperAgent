"""Contracts for a private, short-lived uploaded-paper workspace.

The workspace deliberately stores references to parsed paper artifacts instead of
duplicating PDF text.  A promoted/library paper is invalid here: callers must
query the long-term library through its normal corpus scope.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


def _workspace_id() -> str:
    return f"tmpws_{uuid4().hex[:16]}"


class TemporaryPaperRef(BaseModel):
    """An unpromoted paper and the artifacts needed to retrieve it."""

    paper_id: str = Field(min_length=1, max_length=128)
    ingestion_task_id: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=1_000)
    paper_path: str | None = Field(default=None, max_length=4_096)
    chunks_path: str | None = Field(default=None, max_length=4_096)
    promoted_to_library: bool = False

    @model_validator(mode="after")
    def reject_promoted_papers(self) -> "TemporaryPaperRef":
        if self.promoted_to_library:
            raise ValueError("promoted papers must not be placed in a temporary workspace")
        return self


class TemporaryPaperWorkspace(BaseModel):
    """A user/session-isolated workspace containing only newly uploaded papers."""

    workspace_id: str = Field(default_factory=_workspace_id, min_length=7, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    papers: list[TemporaryPaperRef] = Field(default_factory=list, max_length=20)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(hours=24))

    @field_validator("workspace_id", "user_id", "session_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("identifiers may contain only letters, numbers, '_' and '-'")
        return value

    @model_validator(mode="after")
    def validate_papers_and_expiry(self) -> "TemporaryPaperWorkspace":
        paper_ids = [paper.paper_id for paper in self.papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("temporary workspace paper_ids must be unique")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        instant = now or utc_now()
        if instant.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return self.expires_at <= instant

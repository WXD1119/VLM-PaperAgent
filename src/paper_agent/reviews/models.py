from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from paper_agent.agents import ReviewExecutionResult


class ReviewArtifact(BaseModel):
    """A private, retrievable output of a confirmed review run."""

    review_id: str = Field(default_factory=lambda: f"review_{uuid4().hex[:16]}")
    user_id: str
    execution: ReviewExecutionResult
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

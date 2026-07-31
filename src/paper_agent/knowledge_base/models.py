from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LibraryUpdatePhase(StrEnum):
    VALIDATE = "validate"
    INDEX = "index"
    GRAPH = "graph"
    METADATA = "metadata"


class LibraryUpdateStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"


class LibraryUpdate(BaseModel):
    update_id: str = Field(default_factory=lambda: f"update_{uuid4().hex[:16]}")
    paper_id: str
    content_sha256: str
    parser_version: str
    embedding_model: str
    ingestion_task_id: str | None = None
    status: LibraryUpdateStatus = LibraryUpdateStatus.PENDING
    completed_phases: list[LibraryUpdatePhase] = Field(default_factory=list)
    graph_commit_id: str | None = None
    error: str | None = None
    attempts: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @property
    def idempotency_key(self) -> str:
        return ":".join((self.paper_id, self.content_sha256, self.parser_version, self.embedding_model))


class RebuildKind(StrEnum):
    REPARSE = "reparse"
    REEMBED = "reembed"


class RebuildStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RebuildJob(BaseModel):
    job_id: str = Field(default_factory=lambda: f"rebuild_{uuid4().hex[:16]}")
    paper_id: str
    kind: RebuildKind
    target_parser_version: str | None = None
    target_embedding_model: str | None = None
    status: RebuildStatus = RebuildStatus.PENDING
    error: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

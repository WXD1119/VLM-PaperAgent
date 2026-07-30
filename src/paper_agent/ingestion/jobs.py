"""Durable, file-backed ingestion tasks for the first web upload workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class IngestionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestionTask(BaseModel):
    task_id: str
    filename: str
    source_path: str
    status: IngestionStatus = IngestionStatus.PENDING
    stage: str = "queued"
    paper_id: str | None = None
    paper_path: str | None = None
    chunks_path: str | None = None
    workspace_commit_id: str | None = None
    workspace_path: str | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LocalIngestionTaskStore:
    """One JSON record per task, safe to inspect and recover manually."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.uploads = self.root / "uploads"
        self.tasks = self.root / "tasks"

    def create(self, filename: str, content: bytes) -> IngestionTask:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("only PDF uploads are supported")
        task_id = "ing_" + uuid4().hex[:16]
        self.uploads.mkdir(parents=True, exist_ok=True)
        source_path = self.uploads / f"{task_id}.pdf"
        source_path.write_bytes(content)
        task = IngestionTask(task_id=task_id, filename=Path(filename).name, source_path=str(source_path))
        self.save(task)
        return task

    def list(self, limit: int = 20) -> list[IngestionTask]:
        if not self.tasks.exists():
            return []
        items = [
            IngestionTask.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.tasks.glob("*.json")
        ]
        return sorted(items, key=lambda item: item.created_at, reverse=True)[:limit]

    def load(self, task_id: str) -> IngestionTask:
        path = self._task_path(task_id)
        if not path.exists():
            raise KeyError(f"ingestion task not found: {task_id}")
        return IngestionTask.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, task: IngestionTask) -> IngestionTask:
        self.tasks.mkdir(parents=True, exist_ok=True)
        task.updated_at = datetime.now(UTC)
        self._task_path(task.task_id).write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return task

    def retry(self, task_id: str) -> IngestionTask:
        task = self.load(task_id)
        if task.status == IngestionStatus.RUNNING:
            raise ValueError("cannot retry a running task")
        task.status = IngestionStatus.PENDING
        task.stage = "queued"
        task.error = None
        task.logs.append("retry requested")
        return self.save(task)

    def mark_promoted(
        self,
        task_id: str,
        *,
        workspace_path: str,
        commit_id: str | None,
        message: str,
    ) -> IngestionTask:
        task = self.load(task_id)
        task.workspace_path = workspace_path
        task.workspace_commit_id = commit_id
        task.logs.append(message)
        return self.save(task)

    def _task_path(self, task_id: str) -> Path:
        return self.tasks / f"{task_id}.json"

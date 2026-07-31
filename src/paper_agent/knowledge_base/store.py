from __future__ import annotations

from pathlib import Path

from paper_agent.knowledge_base.models import LibraryUpdate, RebuildJob, RebuildStatus


class LocalKnowledgeBaseStore:
    """Append-safe JSON records for promotion recovery and rebuild scheduling."""

    def __init__(self, root: str | Path = "artifacts/knowledge_base") -> None:
        self.root = Path(root)

    def create_or_load_update(self, update: LibraryUpdate) -> LibraryUpdate:
        for item in self.list_updates(limit=10_000):
            if item.idempotency_key == update.idempotency_key:
                return item
        return self.save_update(update)

    def save_update(self, update: LibraryUpdate) -> LibraryUpdate:
        self._updates.mkdir(parents=True, exist_ok=True)
        self._update_path(update.update_id).write_text(update.model_dump_json(indent=2), encoding="utf-8")
        return update

    def load_update(self, update_id: str) -> LibraryUpdate:
        return LibraryUpdate.model_validate_json(self._update_path(update_id).read_text(encoding="utf-8"))

    def list_updates(self, limit: int = 50) -> list[LibraryUpdate]:
        if not self._updates.exists():
            return []
        values = [LibraryUpdate.model_validate_json(path.read_text(encoding="utf-8")) for path in self._updates.glob("update_*.json")]
        return sorted(values, key=lambda item: item.updated_at, reverse=True)[:limit]

    def enqueue_rebuild(self, job: RebuildJob) -> RebuildJob:
        self._rebuilds.mkdir(parents=True, exist_ok=True)
        self._rebuild_path(job.job_id).write_text(job.model_dump_json(indent=2), encoding="utf-8")
        return job

    def list_rebuilds(self, limit: int = 50) -> list[RebuildJob]:
        if not self._rebuilds.exists():
            return []
        values = [RebuildJob.model_validate_json(path.read_text(encoding="utf-8")) for path in self._rebuilds.glob("rebuild_*.json")]
        return sorted(values, key=lambda item: item.updated_at, reverse=True)[:limit]

    def save_rebuild(self, job: RebuildJob) -> RebuildJob:
        self._rebuilds.mkdir(parents=True, exist_ok=True)
        self._rebuild_path(job.job_id).write_text(job.model_dump_json(indent=2), encoding="utf-8")
        return job

    def claim_next_rebuild(self) -> RebuildJob | None:
        job = next((item for item in self.list_rebuilds(limit=10_000) if item.status == RebuildStatus.PENDING), None)
        if job is None:
            return None
        job.status = RebuildStatus.RUNNING
        return self.save_rebuild(job)

    @property
    def _updates(self) -> Path:
        return self.root / "updates"

    @property
    def _rebuilds(self) -> Path:
        return self.root / "rebuilds"

    def _update_path(self, update_id: str) -> Path:
        if not update_id.startswith("update_") or not update_id.replace("_", "").isalnum():
            raise ValueError("invalid update_id")
        return self._updates / f"{update_id}.json"

    def _rebuild_path(self, job_id: str) -> Path:
        if not job_id.startswith("rebuild_") or not job_id.replace("_", "").isalnum():
            raise ValueError("invalid rebuild job_id")
        return self._rebuilds / f"{job_id}.json"

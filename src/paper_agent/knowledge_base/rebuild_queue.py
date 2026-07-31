from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .models import RebuildJob, RebuildKind, RebuildStatus
from .store import LocalKnowledgeBaseStore


RebuildAction = Callable[[RebuildJob], None]


class RebuildQueue:
    """Small durable queue; the real worker supplies sandboxed reparse/embed actions."""

    def __init__(self, store: LocalKnowledgeBaseStore, *, reparse: RebuildAction, reembed: RebuildAction) -> None:
        self.store = store
        self.actions = {RebuildKind.REPARSE: reparse, RebuildKind.REEMBED: reembed}

    def run_next(self) -> RebuildJob | None:
        job = self.store.claim_next_rebuild()
        if job is None:
            return None
        try:
            self.actions[job.kind](job)
            job.status = RebuildStatus.COMPLETED
            job.error = None
        except Exception as exc:
            job.status = RebuildStatus.FAILED
            job.error = str(exc)
        job.updated_at = datetime.now(UTC).isoformat()
        return self.store.save_rebuild(job)

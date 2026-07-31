from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .models import LibraryUpdate, LibraryUpdatePhase, LibraryUpdateStatus
from .store import LocalKnowledgeBaseStore


PhaseAction = Callable[[LibraryUpdate], str | None]


class LibraryUpdateCoordinator:
    """Runs idempotent promotion phases and persists progress after each phase."""

    PHASES = (LibraryUpdatePhase.VALIDATE, LibraryUpdatePhase.INDEX, LibraryUpdatePhase.GRAPH, LibraryUpdatePhase.METADATA)

    def __init__(self, store: LocalKnowledgeBaseStore, actions: dict[LibraryUpdatePhase, PhaseAction]) -> None:
        self.store = store
        self.actions = actions

    def run(self, update: LibraryUpdate) -> LibraryUpdate:
        current = self.store.create_or_load_update(update)
        if current.status == LibraryUpdateStatus.COMPLETED:
            return current
        current.status = LibraryUpdateStatus.RUNNING
        current.error = None
        current.attempts += 1
        self._save(current)
        try:
            for phase in self.PHASES:
                if phase in current.completed_phases:
                    continue
                action = self.actions.get(phase)
                if action is None:
                    raise RuntimeError(f"missing promotion action: {phase.value}")
                result = action(current)
                if phase == LibraryUpdatePhase.GRAPH and result:
                    current.graph_commit_id = result
                current.completed_phases.append(phase)
                self._save(current)
            current.status = LibraryUpdateStatus.COMPLETED
            self._save(current)
            return current
        except Exception as exc:
            current.status = LibraryUpdateStatus.FAILED
            current.error = str(exc)
            self._save(current)
            raise

    def _save(self, update: LibraryUpdate) -> None:
        update.updated_at = datetime.now(UTC).isoformat()
        self.store.save_update(update)

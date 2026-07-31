"""File-backed lifecycle store for short-lived uploaded-paper workspaces."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

from .models import TemporaryPaperRef, TemporaryPaperWorkspace, utc_now


class TemporaryWorkspaceStore(Protocol):
    def create(
        self,
        *,
        user_id: str,
        session_id: str,
        papers: list[TemporaryPaperRef],
        ttl: timedelta = timedelta(hours=24),
    ) -> TemporaryPaperWorkspace: ...

    def load(self, *, user_id: str, session_id: str, workspace_id: str) -> TemporaryPaperWorkspace: ...

    def delete(self, *, user_id: str, session_id: str, workspace_id: str) -> bool: ...

    def cleanup_expired(self, now: datetime | None = None) -> int: ...


class LocalTemporaryWorkspaceStore:
    """Private JSON persistence with owner checks on every read or deletion.

    The directory layout is ``root/<user>/<session>/<workspace>.json``.  IDs are
    validated by the model before becoming path components; ownership mismatches
    intentionally look identical to a missing workspace.
    """

    def __init__(
        self,
        root: str | Path = "artifacts/temporary_workspaces",
        *,
        on_delete: Callable[[TemporaryPaperWorkspace], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.on_delete = on_delete

    def create(
        self,
        *,
        user_id: str,
        session_id: str,
        papers: list[TemporaryPaperRef],
        ttl: timedelta = timedelta(hours=24),
    ) -> TemporaryPaperWorkspace:
        if ttl <= timedelta(0):
            raise ValueError("temporary workspace ttl must be positive")
        now = utc_now()
        workspace = TemporaryPaperWorkspace(
            user_id=user_id,
            session_id=session_id,
            papers=papers,
            created_at=now,
            updated_at=now,
            expires_at=now + ttl,
        )
        self._write(workspace)
        return workspace

    def load(self, *, user_id: str, session_id: str, workspace_id: str) -> TemporaryPaperWorkspace:
        path = self._path(user_id, session_id, workspace_id)
        if not path.exists():
            raise KeyError(workspace_id)
        workspace = TemporaryPaperWorkspace.model_validate_json(path.read_text(encoding="utf-8"))
        if workspace.user_id != user_id or workspace.session_id != session_id:
            raise KeyError(workspace_id)
        if workspace.is_expired():
            self._remove(path, workspace)
            raise KeyError(workspace_id)
        return workspace

    def delete(self, *, user_id: str, session_id: str, workspace_id: str) -> bool:
        path = self._path(user_id, session_id, workspace_id)
        if not path.exists():
            return False
        workspace = TemporaryPaperWorkspace.model_validate_json(path.read_text(encoding="utf-8"))
        if workspace.user_id != user_id or workspace.session_id != session_id:
            return False
        self._remove(path, workspace)
        return True

    def cleanup_expired(self, now: datetime | None = None) -> int:
        if not self.root.exists():
            return 0
        instant = now or utc_now()
        if instant.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        removed = 0
        for path in self.root.glob("*/*/*.json"):
            try:
                workspace = TemporaryPaperWorkspace.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A malformed file is not owned by this lifecycle manager.
                continue
            if workspace.is_expired(instant):
                self._remove(path, workspace)
                removed += 1
        return removed

    def _write(self, workspace: TemporaryPaperWorkspace) -> None:
        path = self._path(workspace.user_id, workspace.session_id, workspace.workspace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(workspace.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

    def _remove(self, path: Path, workspace: TemporaryPaperWorkspace) -> None:
        path.unlink()
        self._prune_empty_parents(path)
        if self.on_delete:
            self.on_delete(workspace)

    def _path(self, user_id: str, session_id: str, workspace_id: str) -> Path:
        # Model validation centralises both path-component and identifier rules.
        checked = TemporaryPaperWorkspace(user_id=user_id, session_id=session_id, workspace_id=workspace_id)
        return self.root / checked.user_id / checked.session_id / f"{checked.workspace_id}.json"

    def _prune_empty_parents(self, path: Path) -> None:
        for parent in (path.parent, path.parent.parent):
            if parent == self.root or not parent.exists():
                continue
            try:
                parent.rmdir()
            except OSError:
                break

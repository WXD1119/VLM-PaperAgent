from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """Short-term working memory for the current agent run.

    This state helps a conversational agent avoid repeatedly asking for the same
    operational context. It is not paper knowledge and must not be written into the
    paper knowledge graph.
    """

    current_workspace_id: str | None = None
    current_paper_id: str | None = None
    last_query: str | None = None
    last_answer_path: str | None = None
    active_task: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMemoryStore:
    def __init__(self, path: str | Path = "artifacts/memory/session.json") -> None:
        self.path = Path(path)

    def load(self) -> SessionState:
        if not self.path.exists():
            return SessionState()
        return SessionState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: SessionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def update(self, **changes: Any) -> SessionState:
        state = self.load()
        updated = state.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self.save(updated)
        return updated

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

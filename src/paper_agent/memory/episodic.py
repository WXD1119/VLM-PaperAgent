import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """记录 Agent 使用过程的追加式事件记忆。"""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:16]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemoryStore:
    def __init__(self, path: str | Path = "artifacts/memory/episodes.jsonl") -> None:
        self.path = Path(path)

    def append(self, episode: Episode) -> Episode:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return episode

    def log(
        self,
        event_type: str,
        summary: str,
        payload: dict[str, Any] | None = None,
    ) -> Episode:
        return self.append(
            Episode(event_type=event_type, summary=summary, payload=payload or {})
        )

    def list(self, limit: int | None = None) -> list[Episode]:
        if not self.path.exists():
            return []
        episodes = [
            Episode.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is None:
            return episodes
        return episodes[-limit:]

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=lambda: f"turn_{uuid4().hex[:16]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlidingWindowResult(BaseModel):
    active_window: list[ConversationTurn] = Field(default_factory=list)
    overflow: list[ConversationTurn] = Field(default_factory=list)


class SlidingWindowMemory:
    """Keep recent turns active and return old turns for summarization."""

    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns

    def split(self, turns: list[ConversationTurn]) -> SlidingWindowResult:
        if len(turns) <= self.max_turns:
            return SlidingWindowResult(active_window=turns, overflow=[])
        return SlidingWindowResult(
            active_window=turns[-self.max_turns :],
            overflow=turns[: -self.max_turns],
        )


class ConversationSummary(BaseModel):
    summary_id: str = Field(default_factory=lambda: f"sum_{uuid4().hex[:16]}")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_turn_ids: list[str] = Field(default_factory=list)
    summary: str
    key_entities: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        parts = [
            self.summary,
            " ".join(self.key_entities),
            " ".join(self.paper_ids),
            " ".join(self.decisions),
            " ".join(self.unresolved_questions),
        ]
        return "\n".join(part for part in parts if part.strip())


class HeuristicConversationSummarizer:
    """Deterministic summarizer used for tests and offline memory compression.

    A production deployment can replace this with an LLM summarizer while preserving the
    same ConversationSummary schema.
    """

    _PAPER_ID = re.compile(r"paper_[A-Za-z0-9]+")
    _ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9-]{2,}\b")

    def summarize(
        self,
        turns: list[ConversationTurn],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSummary | None:
        if not turns:
            return None
        text = " ".join(turn.content.strip() for turn in turns if turn.content.strip())
        paper_ids = sorted(
            {
                *self._PAPER_ID.findall(text),
                *[
                    str(turn.metadata["paper_id"])
                    for turn in turns
                    if turn.metadata.get("paper_id")
                ],
            }
        )
        key_entities = sorted(set(self._ENTITY.findall(text)))[:12]
        decisions = [
            turn.content.strip()
            for turn in turns
            if _looks_like_decision(turn.content)
        ][:8]
        unresolved = [
            turn.content.strip()
            for turn in turns
            if turn.content.strip().endswith(("?", "？"))
        ][:8]
        return ConversationSummary(
            source_turn_ids=[turn.turn_id for turn in turns],
            summary=_compact_text(text, limit=600),
            key_entities=key_entities,
            paper_ids=paper_ids,
            decisions=decisions,
            unresolved_questions=unresolved,
            metadata=metadata or {},
        )


class SummaryMemoryStore:
    """Append-only summary memory store, separate from the Paper KG."""

    def __init__(self, path: str | Path = "artifacts/memory/summaries.jsonl") -> None:
        self.path = Path(path)

    def append(self, summary: ConversationSummary) -> ConversationSummary:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return summary

    def list(self, limit: int | None = None) -> list[ConversationSummary]:
        if not self.path.exists():
            return []
        summaries = [
            ConversationSummary.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is None:
            return summaries
        return summaries[-limit:]


class SummaryCursor(BaseModel):
    last_summarized_event_id: str | None = None
    last_summary_id: str | None = None
    updated_at: str | None = None


class SummaryCursorStore:
    """Checkpoint for summary compression progress."""

    def __init__(self, path: str | Path = "artifacts/memory/summary_cursor.json") -> None:
        self.path = Path(path)

    def load(self) -> SummaryCursor:
        if not self.path.exists():
            return SummaryCursor()
        return SummaryCursor.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, cursor: SummaryCursor) -> SummaryCursor:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(cursor.model_dump_json(indent=2), encoding="utf-8")
        return cursor

    def update(self, *, event_id: str, summary_id: str) -> SummaryCursor:
        return self.save(
            SummaryCursor(
                last_summarized_event_id=event_id,
                last_summary_id=summary_id,
                updated_at=datetime.now(UTC).isoformat(),
            )
        )


class SummaryCompressionResult(BaseModel):
    total_events: int
    pending_events: int
    active_window_events: int
    overflow_events: int
    summary: ConversationSummary | None = None
    cursor: SummaryCursor
    written: bool = False


def summarize_new_episodes(
    episodes,
    *,
    window_size: int,
    cursor: SummaryCursor | None = None,
    summarizer: HeuristicConversationSummarizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> SummaryCompressionResult:
    """Summarize only episodes after the cursor.

    Compression is triggered only when pending events exceed the sliding-window size.
    """

    cursor = cursor or SummaryCursor()
    pending = _episodes_after_cursor(episodes, cursor.last_summarized_event_id)
    turns = [
        ConversationTurn(
            turn_id=episode.event_id,
            timestamp=episode.timestamp,
            role="event",
            content=episode.summary,
            metadata=episode.payload,
        )
        for episode in pending
    ]
    window = SlidingWindowMemory(window_size).split(turns)
    summary = (summarizer or HeuristicConversationSummarizer()).summarize(
        window.overflow,
        metadata={
            **(metadata or {}),
            "window_size": window_size,
            "active_window_turns": len(window.active_window),
            "cursor_event_id": cursor.last_summarized_event_id,
        },
    )
    return SummaryCompressionResult(
        total_events=len(episodes),
        pending_events=len(pending),
        active_window_events=len(window.active_window),
        overflow_events=len(window.overflow),
        summary=summary,
        cursor=cursor,
        written=False,
    )


def compress_episode_memory(
    episode_store,
    summary_store: SummaryMemoryStore,
    cursor_store: SummaryCursorStore,
    *,
    window_size: int = 6,
    metadata: dict[str, Any] | None = None,
) -> SummaryCompressionResult:
    """Compress overflow episodes and advance the checkpoint atomically enough for local use.

    No compression happens until pending events exceed ``window_size``. The summary is
    written to summary memory, while the Paper KG remains untouched.
    """

    episodes = episode_store.list()
    cursor = cursor_store.load()
    result = summarize_new_episodes(
        episodes,
        window_size=window_size,
        cursor=cursor,
        metadata=metadata,
    )
    if result.summary is None:
        return result
    summary_store.append(result.summary)
    pending = _episodes_after_cursor(episodes, cursor.last_summarized_event_id)
    overflow_count = result.overflow_events
    summarized_events = pending[:overflow_count]
    if summarized_events:
        next_cursor = cursor_store.update(
            event_id=summarized_events[-1].event_id,
            summary_id=result.summary.summary_id,
        )
    else:
        next_cursor = cursor
    return result.model_copy(update={"cursor": next_cursor, "written": True})


class TextEmbedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingTextEmbedder:
    """Dependency-free embedding for deterministic local summary recall tests."""

    _TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, count in Counter(self._tokens(text)).items():
            vector[hash(token) % self.dimension] += float(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        return [token.lower() for token in self._TOKEN.findall(text)]


class SummarySearchHit(BaseModel):
    summary: ConversationSummary
    score: float


class InMemorySummaryVectorStore:
    """Small vector store for summary memory.

    This is intentionally independent from the paper chunk vector collection. In a
    production deployment, the same schema can be backed by Chroma/FAISS/pgvector.
    """

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self.embedder = embedder or HashingTextEmbedder()
        self._records: list[tuple[ConversationSummary, list[float]]] = []

    def add(self, summaries: list[ConversationSummary]) -> None:
        for summary in summaries:
            self._records.append((summary, self.embedder.embed(summary.embedding_text)))

    def search(self, query: str, top_k: int = 5) -> list[SummarySearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self.embedder.embed(query)
        hits = [
            SummarySearchHit(summary=summary, score=_cosine(query_vector, vector))
            for summary, vector in self._records
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.summary.summary_id))
        return [hit for hit in hits if hit.score > 0][:top_k]


def build_summary_vector_store(
    summaries: list[ConversationSummary],
    *,
    embedder: TextEmbedder | None = None,
) -> InMemorySummaryVectorStore:
    store = InMemorySummaryVectorStore(embedder)
    store.add(summaries)
    return store


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _looks_like_decision(text: str) -> bool:
    lowered = text.lower()
    markers = ["decide", "decided", "choose", "chosen", "should", "采用", "决定", "选择", "需要"]
    return any(marker in lowered for marker in markers)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _episodes_after_cursor(episodes, event_id: str | None):
    if event_id is None:
        return list(episodes)
    for index, episode in enumerate(episodes):
        if episode.event_id == event_id:
            return list(episodes[index + 1 :])
    return list(episodes)

"""面向证据问答工作流的轻量级结构化追踪。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """一次状态图节点或外部依赖调用的脱敏执行记录。"""

    name: str
    status: str
    elapsed_ms: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TraceRecord(BaseModel):
    """一次问答请求的完整运行轨迹；不保存原始问题与论文正文。"""

    trace_id: str
    started_at: str
    completed_at: str | None = None
    status: str = "running"
    query_sha256: str
    runtime: str
    user_id: str | None = None
    session_id: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TraceRecorder:
    """在内存中累计追踪事件，完成后由 TraceStore 一次性落盘。"""

    def __init__(
        self,
        *,
        query: str,
        runtime: str,
        user_id: str | None = None,
        session_id: str | None = None,
        event_listener: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.record = TraceRecord(
            trace_id=f"tr_{uuid4().hex[:16]}",
            started_at=datetime.now(UTC).isoformat(),
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            runtime=runtime,
            user_id=user_id,
            session_id=session_id,
        )
        self.event_listener = event_listener

    def run(self, name: str, callback, *, attributes: dict[str, Any] | None = None):
        """记录回调耗时；异常保留后继续向上抛出，避免隐藏主流程失败。"""

        started_at = perf_counter()
        self._notify(TraceEvent(name=name, status="running", elapsed_ms=0, attributes=attributes or {}))
        try:
            value = callback()
        except Exception as exc:
            self.record_event(
                name,
                status="error",
                elapsed_ms=_elapsed_ms(started_at),
                attributes=attributes,
                error=_safe_error(exc),
            )
            raise
        self.record_event(
            name,
            status="success",
            elapsed_ms=_elapsed_ms(started_at),
            attributes=attributes,
        )
        return value

    def record_event(
        self,
        name: str,
        *,
        status: str,
        elapsed_ms: int,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.record.events.append(
            TraceEvent(name=name, status=status, elapsed_ms=elapsed_ms, attributes=attributes or {}, error=error)
        )
        self._notify(self.record.events[-1])

    def complete(
        self,
        *,
        status: str,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TraceRecord:
        self.record.status = status
        self.record.completed_at = datetime.now(UTC).isoformat()
        self.record.attributes.update(attributes or {})
        self.record.error = error
        return self.record

    def _notify(self, event: TraceEvent) -> None:
        """监听器仅用于实时展示；监听器异常不能影响问答与追踪落盘。"""

        if self.event_listener is None:
            return
        try:
            self.event_listener(event)
        except Exception:
            pass


class JsonlTraceStore:
    """追加式 JSONL 追踪存储，适合作为本地开发和故障回放的最小实现。"""

    def __init__(self, path: str | Path = "artifacts/traces/traces.jsonl") -> None:
        self.path = Path(path)

    def append(self, trace: TraceRecord) -> TraceRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json() + "\n")
        return trace

    def list(self, limit: int = 50, *, user_id: str | None = None) -> list[TraceRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            return []
        rows = [
            TraceRecord.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if user_id is not None:
            rows = [trace for trace in rows if trace.user_id == user_id]
        return list(reversed(rows[-limit:]))

    def load(self, trace_id: str, *, user_id: str | None = None) -> TraceRecord:
        for trace in self.list(limit=10_000, user_id=user_id):
            if trace.trace_id == trace_id:
                return trace
        raise KeyError(trace_id)


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _safe_error(error: Exception, limit: int = 500) -> str:
    """避免把异常中的大段输入、密钥或模型输出直接写入 Trace。"""

    text = f"{type(error).__name__}: {error}".replace("\n", " ")
    return text[:limit]

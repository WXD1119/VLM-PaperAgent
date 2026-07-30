from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any


class ResourceAccess(StrEnum):
    READ = "read"
    WRITE = "write"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN_TOOL = "unknown_tool"


ToolHandler = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: ToolHandler
    timeout_s: float = 10.0
    resources: frozenset[str] = field(default_factory=frozenset)
    access: ResourceAccess = ResourceAccess.READ
    description: str = ""
    max_retries: int = 0
    retry_backoff_s: float = 0.0
    input_schema: str = "object"
    output_schema: str = "object"
    required_scopes: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolStatus
    output: Any = None
    error: str | None = None
    elapsed_ms: int = 0
    attempts: int = 0


class ToolOrchestrator:
    """按超时和简单资源冲突规则运行已注册工具。

    冲突规则：读/读调用可并行；任何写入同一资源的调用都会与该资源的其他读写调用串行。
    """

    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        if tool.timeout_s <= 0:
            raise ValueError("tool timeout must be positive")
        if tool.max_retries < 0:
            raise ValueError("tool max_retries cannot be negative")
        if tool.retry_backoff_s < 0:
            raise ValueError("tool retry_backoff_s cannot be negative")
        self._tools[tool.name] = tool

    def plan_batches(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        batches: list[list[ToolCall]] = []
        batch_specs: list[list[ToolSpec | None]] = []
        for call in calls:
            spec = self._tools.get(call.tool_name)
            placed = False
            for index, specs in enumerate(batch_specs):
                if all(not self._conflicts(spec, existing) for existing in specs):
                    batches[index].append(call)
                    specs.append(spec)
                    placed = True
                    break
            if not placed:
                batches.append([call])
                batch_specs.append([spec])
        return batches

    def run_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        results: list[ToolResult] = []
        pending = list(calls)
        completed: set[str] = set()
        while pending:
            ready = [call for call in pending if set(call.depends_on) <= completed]
            if not ready:
                results.extend(
                    ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status=ToolStatus.ERROR,
                        error=f"unresolved tool dependencies: {', '.join(call.depends_on)}",
                    )
                    for call in pending
                )
                break
            batch = self._first_ready_batch(ready)
            batch_results = self._run_batch(batch)
            results.extend(batch_results)
            completed.update(result.call_id for result in batch_results)
            pending = [call for call in pending if call.call_id not in completed]
        return [next(result for result in results if result.call_id == call.call_id) for call in calls]

    def _first_ready_batch(self, calls: list[ToolCall]) -> list[ToolCall]:
        batches = self.plan_batches(calls)
        return batches[0] if batches else []

    def _run_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        if not calls:
            return []
        results_by_id: dict[str, ToolResult] = {}
        executable: list[tuple[ToolCall, ToolSpec]] = []
        for call in calls:
            spec = self._tools.get(call.tool_name)
            if spec is None:
                results_by_id[call.call_id] = ToolResult(
                    call_id=call.call_id,
                    tool_name=call.tool_name,
                    status=ToolStatus.UNKNOWN_TOOL,
                    error=f"unknown tool: {call.tool_name}",
                )
            else:
                executable.append((call, spec))

        if executable:
            executor = ThreadPoolExecutor(max_workers=len(executable))
            try:
                future_by_call = {
                    executor.submit(self._execute_with_retries, call, spec): (call, spec, monotonic())
                    for call, spec in executable
                }
                for future, (call, spec, started_at) in future_by_call.items():
                    try:
                        output, attempts = future.result(timeout=spec.timeout_s * (spec.max_retries + 1))
                        status = ToolStatus.SUCCESS
                        error = None
                    except TimeoutError:
                        future.cancel()
                        output = None
                        status = ToolStatus.TIMEOUT
                        error = f"tool timed out after {spec.timeout_s:.3f}s"
                        attempts = spec.max_retries + 1
                    except Exception as exc:  # 工具边界：隔离单个工具失败
                        output = None
                        status = ToolStatus.ERROR
                        error = str(exc)
                        attempts = spec.max_retries + 1
                    results_by_id[call.call_id] = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status=status,
                        output=output,
                        error=error,
                        elapsed_ms=int((monotonic() - started_at) * 1000),
                        attempts=attempts,
                    )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        return [results_by_id[call.call_id] for call in calls]

    @staticmethod
    def _execute_with_retries(call: ToolCall, spec: ToolSpec) -> tuple[Any, int]:
        import time

        attempts = 0
        while True:
            attempts += 1
            try:
                return spec.handler(call.args), attempts
            except Exception:
                if attempts > spec.max_retries:
                    raise
                if spec.retry_backoff_s:
                    time.sleep(spec.retry_backoff_s * (2 ** (attempts - 1)))

    def _conflicts(self, left: ToolSpec | None, right: ToolSpec | None) -> bool:
        if left is None or right is None:
            return False
        shared = left.resources & right.resources
        if not shared:
            return False
        return left.access == ResourceAccess.WRITE or right.access == ResourceAccess.WRITE

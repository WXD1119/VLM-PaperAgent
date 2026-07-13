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


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: ToolStatus
    output: Any = None
    error: str | None = None
    elapsed_ms: int = 0


class ToolOrchestrator:
    """Run registered tools with timeout and simple resource-conflict scheduling.

    Conflict policy: read/read calls can share a batch, but any write touching the same
    resource is serialized away from other read/write calls on that resource.
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
        for batch in self.plan_batches(calls):
            results.extend(self._run_batch(batch))
        return results

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
            with ThreadPoolExecutor(max_workers=len(executable)) as executor:
                future_by_call = {
                    executor.submit(spec.handler, call.args): (call, spec, monotonic())
                    for call, spec in executable
                }
                for future, (call, spec, started_at) in future_by_call.items():
                    try:
                        output = future.result(timeout=spec.timeout_s)
                        status = ToolStatus.SUCCESS
                        error = None
                    except TimeoutError:
                        future.cancel()
                        output = None
                        status = ToolStatus.TIMEOUT
                        error = f"tool timed out after {spec.timeout_s:.3f}s"
                    except Exception as exc:  # tool boundary: isolate individual failures
                        output = None
                        status = ToolStatus.ERROR
                        error = str(exc)
                    results_by_id[call.call_id] = ToolResult(
                        call_id=call.call_id,
                        tool_name=call.tool_name,
                        status=status,
                        output=output,
                        error=error,
                        elapsed_ms=int((monotonic() - started_at) * 1000),
                    )

        return [results_by_id[call.call_id] for call in calls]

    def _conflicts(self, left: ToolSpec | None, right: ToolSpec | None) -> bool:
        if left is None or right is None:
            return False
        shared = left.resources & right.resources
        if not shared:
            return False
        return left.access == ResourceAccess.WRITE or right.access == ResourceAccess.WRITE

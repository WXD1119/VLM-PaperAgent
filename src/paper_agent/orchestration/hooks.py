"""Small deterministic lifecycle hooks for policy and observability."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class HookEvent(BaseModel):
    phase: str
    node: str
    attributes: dict[str, Any] = Field(default_factory=dict)


Hook = Callable[[HookEvent], None]


class WorkflowHooks:
    """Observers only: hook errors never change graph control flow."""

    def __init__(self, hooks: dict[str, list[Hook]] | None = None) -> None:
        self._hooks = hooks or {}

    def emit(self, phase: str, node: str, **attributes: Any) -> None:
        event = HookEvent(phase=phase, node=node, attributes=attributes)
        for hook in self._hooks.get(phase, []):
            try:
                hook(event)
            except Exception:
                # Hooks cannot become an untrusted alternate control plane.
                continue

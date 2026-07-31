"""Shared contracts and safeguards for the research-workspace runtime."""

from .context import ContextManager, ContextSelection
from .contracts import (
    AgentDispatch,
    AgentName,
    ContextBudget,
    CorpusScope,
    HandoffPacket,
    ResearchSession,
    ScopeResolution,
    resolve_scope,
)
from .hooks import HookEvent, WorkflowHooks
from .router import ResearchRouteDecision, ResearchRouteRequest, ResearchRouter

__all__ = [
    "AgentDispatch",
    "AgentName",
    "ContextBudget",
    "ContextManager",
    "ContextSelection",
    "CorpusScope",
    "HandoffPacket",
    "HookEvent",
    "ResearchSession",
    "ResearchRouteDecision",
    "ResearchRouteRequest",
    "ResearchRouter",
    "ScopeResolution",
    "WorkflowHooks",
    "resolve_scope",
]

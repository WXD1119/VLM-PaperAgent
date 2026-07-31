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
from .policy import GovernedWorkflowHooks, GovernanceCounters, GovernanceViolation, WorkflowGovernance, WorkflowLimits
from .router import ResearchRouteDecision, ResearchRouteRequest, ResearchRouter

__all__ = [
    "AgentDispatch",
    "AgentName",
    "ContextBudget",
    "ContextManager",
    "ContextSelection",
    "CorpusScope",
    "HandoffPacket",
    "GovernedWorkflowHooks",
    "GovernanceCounters",
    "GovernanceViolation",
    "HookEvent",
    "ResearchSession",
    "ResearchRouteDecision",
    "ResearchRouteRequest",
    "ResearchRouter",
    "ScopeResolution",
    "WorkflowHooks",
    "WorkflowGovernance",
    "WorkflowLimits",
    "resolve_scope",
]

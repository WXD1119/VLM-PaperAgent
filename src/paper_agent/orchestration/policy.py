"""Enforceable pre-flight controls for research-workflow calls.

``WorkflowHooks`` deliberately remains observational.  This module wraps it with
an admission controller so that policy failures stop a call *before* a model or
tool is invoked.  The controller is deterministic and does not trust values
suggested by an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from paper_agent.security import ActorContext, ToolAuthorizationPolicy

from .contracts import ContextBudget, CorpusScope, ScopeResolution
from .hooks import WorkflowHooks


class GovernanceViolation(PermissionError):
    """A workflow operation was rejected before execution."""


class WorkflowLimits(BaseModel):
    """Hard per-run admission limits, independent from provider quotas."""

    max_model_calls: int = Field(default=8, ge=0)
    max_tool_calls: int = Field(default=48, ge=0)
    max_network_calls: int = Field(default=0, ge=0)
    max_model_input_tokens: int = Field(default=24_000, ge=0)
    max_model_output_tokens: int = Field(default=8_000, ge=0)


@dataclass
class GovernanceCounters:
    model_calls: int = 0
    tool_calls: int = 0
    network_calls: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0


_NETWORK_TOOLS = frozenset({"web_discover", "web_search", "fetch_web_candidates"})
_SPECIAL_TOOL_SCOPES: dict[str, frozenset[str]] = {
    "web_discover": frozenset({"web:discover"}),
    "web_search": frozenset({"web:discover"}),
    "fetch_web_candidates": frozenset({"web:discover"}),
    "persist_review_artifact": frozenset({"artifact:write"}),
    "write_review_artifact": frozenset({"artifact:write"}),
    "export_review": frozenset({"artifact:export"}),
}


@dataclass
class WorkflowGovernance:
    """Stateful, fail-closed admission control for one workflow invocation."""

    actor: ActorContext
    scope: ScopeResolution
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    limits: WorkflowLimits | None = None
    web_expansion_confirmed: bool = False
    authorizer: ToolAuthorizationPolicy = field(default_factory=ToolAuthorizationPolicy)
    counters: GovernanceCounters = field(default_factory=GovernanceCounters)

    def __post_init__(self) -> None:
        if self.limits is None:
            # ContextBudget describes what a node may pack into a model context.
            # Provider-facing output allowance remains explicit and conservative.
            self.limits = WorkflowLimits(
                max_model_input_tokens=self.context_budget.total_tokens,
                max_model_output_tokens=self.context_budget.output_reserve_tokens,
            )

    def before_tool(self, name: str, **attributes: Any) -> None:
        """Authorize a named, allow-listed tool and account for its use."""

        self._authorize_tool(name)
        self._enforce_paper_scope(attributes.get("paper_ids"))
        is_network = name in _NETWORK_TOOLS or bool(attributes.get("network"))
        if is_network:
            self._enforce_network(name)
        assert self.limits is not None
        if self.counters.tool_calls >= self.limits.max_tool_calls:
            raise GovernanceViolation("tool call budget exhausted")
        self.counters.tool_calls += 1
        if is_network:
            self.counters.network_calls += 1

    def before_model(self, name: str, **attributes: Any) -> None:
        """Reserve declared input/output tokens before sending a model request."""

        input_tokens = _non_negative_int(attributes.get("input_tokens", 0), "input_tokens")
        output_tokens = _non_negative_int(attributes.get("output_tokens", 0), "output_tokens")
        self._enforce_paper_scope(attributes.get("paper_ids"))
        assert self.limits is not None
        if self.counters.model_calls >= self.limits.max_model_calls:
            raise GovernanceViolation("model call budget exhausted")
        if self.counters.model_input_tokens + input_tokens > self.limits.max_model_input_tokens:
            raise GovernanceViolation("model input token budget exhausted")
        if self.counters.model_output_tokens + output_tokens > self.limits.max_model_output_tokens:
            raise GovernanceViolation("model output token budget exhausted")
        self.counters.model_calls += 1
        self.counters.model_input_tokens += input_tokens
        self.counters.model_output_tokens += output_tokens

    def _authorize_tool(self, name: str) -> None:
        required = _SPECIAL_TOOL_SCOPES.get(name)
        if required is None:
            try:
                self.authorizer.authorize(self.actor, name)
            except PermissionError as exc:
                raise GovernanceViolation(str(exc)) from exc
            return
        if not required <= self.actor.scopes:
            raise GovernanceViolation(f"missing scope for {name}: {', '.join(sorted(required))}")

    def _enforce_network(self, name: str) -> None:
        assert self.limits is not None
        if self.scope.scope != CorpusScope.WEB_EXPANSION or not self.web_expansion_confirmed:
            raise GovernanceViolation(f"network tool {name} requires confirmed web_expansion scope")
        if self.counters.network_calls >= self.limits.max_network_calls:
            raise GovernanceViolation("network call budget exhausted")

    def _enforce_paper_scope(self, paper_ids: Any) -> None:
        if paper_ids is None:
            return
        if isinstance(paper_ids, str):
            values = [paper_ids]
        else:
            try:
                values = list(paper_ids)
            except TypeError as exc:
                raise GovernanceViolation("paper_ids must be a string or iterable of strings") from exc
        denied = [str(paper_id) for paper_id in values if not self.scope.allows(str(paper_id))]
        if denied:
            raise GovernanceViolation(f"paper IDs outside corpus scope: {', '.join(denied)}")


class GovernedWorkflowHooks(WorkflowHooks):
    """A hook facade whose ``before_*`` phases are mandatory policy gates."""

    def __init__(self, governance: WorkflowGovernance, hooks: dict[str, list] | None = None) -> None:
        super().__init__(hooks=hooks)
        self.governance = governance

    def emit(self, phase: str, node: str, **attributes: Any) -> None:
        if phase == "before_tool":
            self.governance.before_tool(node, **attributes)
        elif phase == "before_model":
            self.governance.before_model(node, **attributes)
        super().emit(phase, node, **attributes)


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GovernanceViolation(f"{label} must be a non-negative integer")
    return value

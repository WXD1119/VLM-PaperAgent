import pytest

from paper_agent.orchestration import (
    ContextBudget,
    CorpusScope,
    GovernedWorkflowHooks,
    GovernanceViolation,
    WorkflowGovernance,
    WorkflowLimits,
    resolve_scope,
)
from paper_agent.security import ActorContext


def _governance(*, scopes=frozenset({"paper:read"}), scope=CorpusScope.ACTIVE_PAPER_ONLY, **kwargs):
    return WorkflowGovernance(
        actor=ActorContext(user_id="wxd", session_id="s1", scopes=scopes),
        scope=resolve_scope(scope, active_paper_id="paper-1", web_expansion_confirmed=True),
        **kwargs,
    )


def test_hook_rejects_tool_before_observers_or_execution():
    observed = []
    hooks = GovernedWorkflowHooks(
        _governance(),
        hooks={"before_tool": [lambda event: observed.append(event.node)]},
    )
    with pytest.raises(GovernanceViolation, match="missing scope"):
        hooks.emit("before_tool", "persist_review_artifact")
    assert observed == []


def test_tool_access_is_hard_scoped_to_active_paper():
    governance = _governance()
    governance.before_tool("retrieve_evidence", paper_ids=["paper-1"])
    with pytest.raises(GovernanceViolation, match="outside corpus scope"):
        governance.before_tool("retrieve_evidence", paper_ids=["paper-other"])


def test_model_budget_is_reserved_before_the_model_call():
    governance = _governance(
        context_budget=ContextBudget(total_tokens=2_000, system_tokens=100, state_tokens=100, memory_tokens=100, evidence_tokens=100, draft_tokens=100, output_reserve_tokens=256),
        limits=WorkflowLimits(max_model_calls=1, max_model_input_tokens=10, max_model_output_tokens=5),
    )
    governance.before_model("generate_answer", input_tokens=8, output_tokens=5)
    with pytest.raises(GovernanceViolation, match="model call budget exhausted"):
        governance.before_model("generate_answer", input_tokens=1, output_tokens=0)
    assert governance.counters.model_calls == 1


def test_network_needs_web_scope_confirmation_capability_and_quota():
    actor_scopes = frozenset({"paper:read", "web:discover"})
    governance = _governance(
        scopes=actor_scopes,
        scope=CorpusScope.WEB_EXPANSION,
        web_expansion_confirmed=True,
        limits=WorkflowLimits(max_network_calls=1),
    )
    governance.before_tool("web_discover")
    with pytest.raises(GovernanceViolation, match="network call budget exhausted"):
        governance.before_tool("web_discover")


def test_network_is_rejected_without_explicit_confirmation():
    governance = _governance(
        scopes=frozenset({"paper:read", "web:discover"}),
        scope=CorpusScope.WEB_EXPANSION,
        web_expansion_confirmed=False,
        limits=WorkflowLimits(max_network_calls=1),
    )
    with pytest.raises(GovernanceViolation, match="confirmed web_expansion"):
        governance.before_tool("web_discover")

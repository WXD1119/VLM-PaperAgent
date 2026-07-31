from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_agent.evaluation import (
    RouterDispatchCase,
    ScopeLeakCase,
    evaluate_router_dispatches,
    evaluate_scope_leaks,
)
from paper_agent.orchestration import ResearchRouteRequest
from paper_agent.security import (
    NetworkMode,
    ParserWorkerSandbox,
    TrustedActorResolver,
    VerifiedIdentity,
    WorkerResourceLimits,
)


class _Verifier:
    def __init__(self, identity: VerifiedIdentity) -> None:
        self.identity = identity

    def verify(self, credential: str) -> VerifiedIdentity:
        assert credential == "verified-token"
        return self.identity


def _identity(**changes) -> VerifiedIdentity:
    values = {
        "subject": "reader-1",
        "session_id": "session-1",
        "scopes": frozenset({"paper:read"}),
        "issuer": "https://identity.example",
        "audience": "paper-agent",
        "expires_at": datetime.now(UTC) + timedelta(minutes=5),
    }
    values.update(changes)
    return VerifiedIdentity(**values)


def test_trusted_resolver_requires_verified_matching_unexpired_identity():
    resolver = TrustedActorResolver(
        _Verifier(_identity()), issuer="https://identity.example", audience="paper-agent"
    )
    actor = resolver.resolve("verified-token", requested_session_id="session-1")
    assert actor.user_id == "reader-1"
    assert actor.scopes == frozenset({"paper:read"})

    expired = TrustedActorResolver(
        _Verifier(_identity(expires_at=datetime.now(UTC) - timedelta(seconds=1))),
        issuer="https://identity.example",
        audience="paper-agent",
    )
    with pytest.raises(PermissionError, match="expired"):
        expired.resolve("verified-token")
    with pytest.raises(PermissionError, match="session"):
        resolver.resolve("verified-token", requested_session_id="other-session")


def test_parser_sandbox_rejects_network_and_path_escape(tmp_path: Path):
    sandbox = ParserWorkerSandbox(
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
        scratch_root=tmp_path / "scratch",
    )
    sandbox.validate_job_paths(
        source=tmp_path / "input" / "upload.pdf", output=tmp_path / "output" / "result"
    )
    with pytest.raises(PermissionError, match="source"):
        sandbox.validate_job_paths(source=tmp_path / "outside.pdf", output=tmp_path / "output" / "result")
    with pytest.raises(ValueError, match="allowlist"):
        ParserWorkerSandbox(
            input_root=tmp_path / "a",
            output_root=tmp_path / "b",
            scratch_root=tmp_path / "c",
            network_mode=NetworkMode.ALLOWLIST,
        ).validate()
    with pytest.raises(ValueError, match="cpu_cores"):
        WorkerResourceLimits(cpu_cores=0).validate()


def test_scope_leak_and_router_dispatch_baselines():
    scope_result = evaluate_scope_leaks(
        [
            ScopeLeakCase(case_id="safe", allowed_paper_ids={"p1"}, retrieved_paper_ids=["p1"]),
            ScopeLeakCase(case_id="leak", allowed_paper_ids={"p1"}, retrieved_paper_ids=["p1", "p2"]),
        ]
    )
    assert scope_result.safe_case_rate == 0.5
    assert scope_result.leaked_paper_rate == pytest.approx(1 / 3)
    assert scope_result.cases[1].leaked_paper_ids == ["p2"]

    dispatch_result = evaluate_router_dispatches(
        [
            RouterDispatchCase(
                case_id="review",
                request=ResearchRouteRequest(query="write a survey", mode="review"),
                expected_modes=["review_write"],
                expected_confirmation=True,
            ),
            RouterDispatchCase(
                case_id="compare-needs-selection",
                request=ResearchRouteRequest(query="compare methods"),
                expected_modes=["multi_paper_compare"],
                expected_clarification=True,
            ),
        ]
    )
    assert dispatch_result.exact_dispatch_rate == 1.0
    assert dispatch_result.confirmation_accuracy == 1.0
    assert dispatch_result.clarification_accuracy == 1.0

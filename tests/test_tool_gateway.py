import pytest

from paper_agent.security import ActorContext
from paper_agent.tools import AuthorizedToolGateway, ToolRuntimePolicy


def test_authorized_gateway_executes_allowlisted_read_tool():
    gateway = AuthorizedToolGateway(ActorContext(user_id="wxd", session_id="s1"))
    assert gateway.run("retrieve_evidence", lambda: "ok") == "ok"


def test_authorized_gateway_rejects_write_scope_and_timeout():
    reader = AuthorizedToolGateway(ActorContext(user_id="wxd", session_id="s1"))
    with pytest.raises(PermissionError, match="missing scope"):
        reader.run("promote_paper_to_workspace", lambda: "never")

    slow = AuthorizedToolGateway(
        ActorContext(user_id="wxd", session_id="s1"),
        policies={"retrieve_evidence": ToolRuntimePolicy(timeout_s=0.001)},
    )
    with pytest.raises(RuntimeError, match="timeout"):
        slow.run("retrieve_evidence", lambda: __import__("time").sleep(0.02))

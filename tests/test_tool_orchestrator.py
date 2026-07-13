import time

from paper_agent.tools import (
    ResourceAccess,
    ToolCall,
    ToolOrchestrator,
    ToolSpec,
    ToolStatus,
)


def test_tool_orchestrator_runs_registered_tools_and_preserves_order():
    orchestrator = ToolOrchestrator(
        [
            ToolSpec(
                name="echo",
                handler=lambda args: {"value": args["value"]},
            )
        ]
    )

    results = orchestrator.run_many(
        [
            ToolCall(call_id="c1", tool_name="echo", args={"value": 1}),
            ToolCall(call_id="c2", tool_name="echo", args={"value": 2}),
        ]
    )

    assert [result.call_id for result in results] == ["c1", "c2"]
    assert [result.output["value"] for result in results] == [1, 2]
    assert all(result.status == ToolStatus.SUCCESS for result in results)


def test_tool_orchestrator_serializes_write_conflicts_but_batches_reads():
    read_a = ToolSpec(
        name="read_a",
        handler=lambda args: "a",
        resources=frozenset({"graph"}),
        access=ResourceAccess.READ,
    )
    read_b = ToolSpec(
        name="read_b",
        handler=lambda args: "b",
        resources=frozenset({"graph"}),
        access=ResourceAccess.READ,
    )
    write = ToolSpec(
        name="write",
        handler=lambda args: "w",
        resources=frozenset({"graph"}),
        access=ResourceAccess.WRITE,
    )
    orchestrator = ToolOrchestrator([read_a, read_b, write])

    batches = orchestrator.plan_batches(
        [
            ToolCall(call_id="r1", tool_name="read_a"),
            ToolCall(call_id="r2", tool_name="read_b"),
            ToolCall(call_id="w1", tool_name="write"),
        ]
    )

    assert [[call.call_id for call in batch] for batch in batches] == [["r1", "r2"], ["w1"]]


def test_tool_orchestrator_marks_timeout_without_raising():
    def slow(_args):
        time.sleep(0.05)
        return "done"

    orchestrator = ToolOrchestrator(
        [
            ToolSpec(name="slow", handler=slow, timeout_s=0.001),
            ToolSpec(name="fast", handler=lambda args: "ok"),
        ]
    )

    results = orchestrator.run_many(
        [
            ToolCall(call_id="slow-call", tool_name="slow"),
            ToolCall(call_id="fast-call", tool_name="fast"),
        ]
    )

    assert results[0].status == ToolStatus.TIMEOUT
    assert "timed out" in results[0].error
    assert results[1].status == ToolStatus.SUCCESS
    assert results[1].output == "ok"


def test_tool_orchestrator_isolates_unknown_tools_and_errors():
    def explode(_args):
        raise RuntimeError("boom")

    orchestrator = ToolOrchestrator([ToolSpec(name="explode", handler=explode)])

    results = orchestrator.run_many(
        [
            ToolCall(call_id="bad", tool_name="explode"),
            ToolCall(call_id="missing", tool_name="missing"),
        ]
    )

    assert results[0].status == ToolStatus.ERROR
    assert results[0].error == "boom"
    assert results[1].status == ToolStatus.UNKNOWN_TOOL
    assert results[1].error == "unknown tool: missing"

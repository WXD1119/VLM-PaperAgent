from paper_agent.observability import JsonlTraceStore, TraceRecorder


def test_trace_store_keeps_node_events_and_isolates_users(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    first = TraceRecorder(query="paper question", runtime="langgraph", user_id="alice", session_id="s1")
    first.run("retrieve_evidence", lambda: ["chunk-1"])
    first_trace = store.append(first.complete(status="success", attributes={"evidence_count": 1}))

    second = TraceRecorder(query="another question", runtime="langgraph", user_id="bob", session_id="s2")
    second_trace = store.append(second.complete(status="success"))

    traces = store.list(user_id="alice")
    assert [trace.trace_id for trace in traces] == [first_trace.trace_id]
    assert traces[0].events[0].name == "retrieve_evidence"
    assert traces[0].query_sha256 != "paper question"
    assert store.load(first_trace.trace_id, user_id="alice").status == "success"
    try:
        store.load(second_trace.trace_id, user_id="alice")
    except KeyError:
        pass
    else:
        raise AssertionError("trace must not be readable across users")


def test_trace_recorder_keeps_route_and_evidence_policy_without_raw_query():
    recorder = TraceRecorder(query="解释式(3)的训练目标", runtime="langgraph")
    recorder.record_event(
        "route_decision",
        status="success",
        elapsed_ms=0,
        attributes={"intent": "formula_explanation", "evidence_types": ["equation", "text"]},
    )
    recorder.record_event(
        "evidence_plan",
        status="success",
        elapsed_ms=0,
        attributes={"sub_question_count": 1, "minimum_evidence_count": 2},
    )
    trace = recorder.complete(status="success", attributes={"intent": "formula_explanation"})

    assert trace.query_sha256 != "解释式(3)的训练目标"
    assert [event.name for event in trace.events] == ["route_decision", "evidence_plan"]
    assert trace.events[0].attributes["evidence_types"] == ["equation", "text"]
    assert trace.attributes["intent"] == "formula_explanation"

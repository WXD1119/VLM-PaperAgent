import time

from paper_agent.tools import ResourceAccess, ToolCall, ToolOrchestrator, ToolSpec


def main() -> None:
    orchestrator = ToolOrchestrator(
        [
            ToolSpec(
                name="search_chunks",
                handler=lambda args: {"query": args["query"], "hits": 5},
                resources=frozenset({"paper_index"}),
                access=ResourceAccess.READ,
                description="Read-only retrieval over paper chunks.",
            ),
            ToolSpec(
                name="query_graph",
                handler=lambda args: {"concept": args["concept"], "nodes": 3},
                resources=frozenset({"paper_graph"}),
                access=ResourceAccess.READ,
                description="Read-only graph query.",
            ),
            ToolSpec(
                name="inspect_session",
                handler=lambda args: {"current": args.get("key", "last_query")},
                resources=frozenset({"session_memory"}),
                access=ResourceAccess.READ,
                description="Read short-term session memory.",
            ),
            ToolSpec(
                name="update_session",
                handler=lambda args: {"updated": args["key"]},
                resources=frozenset({"session_memory"}),
                access=ResourceAccess.WRITE,
                description="Write short-term session memory.",
            ),
            ToolSpec(
                name="slow_tool",
                handler=lambda args: time.sleep(args.get("sleep_s", 0.05)),
                timeout_s=0.001,
                resources=frozenset({"remote_service"}),
                access=ResourceAccess.READ,
                description="Demonstrate timeout handling.",
            ),
        ]
    )
    calls = [
        ToolCall("search", "search_chunks", {"query": "Q-Former"}),
        ToolCall("graph", "query_graph", {"concept": "Q-Former"}),
        ToolCall("session-read", "inspect_session", {"key": "last_query"}),
        ToolCall("memory", "update_session", {"key": "last_query"}),
        ToolCall("timeout", "slow_tool", {"sleep_s": 0.05}),
    ]

    print("Batches:")
    for index, batch in enumerate(orchestrator.plan_batches(calls), start=1):
        print(f"- batch {index}: {', '.join(call.call_id for call in batch)}")

    print("\nResults:")
    for result in orchestrator.run_many(calls):
        print(
            f"- {result.call_id}: status={result.status.value} "
            f"elapsed_ms={result.elapsed_ms} error={result.error or '-'} output={result.output}"
        )


if __name__ == "__main__":
    main()

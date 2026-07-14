# Tool Orchestration

M4 introduces a lightweight tool orchestration layer.

It is intentionally separate from the workflow FSM:

- FSM / `Scheduler`: decides which workflow state should run next.
- Tool Orchestrator: decides how concrete tools are executed safely inside a step.

## Why this exists

The paper agent uses several capabilities that behave like tools:

- chunk retrieval;
- graph query;
- session/profile memory update;
- answer generation;
- citation judging;
- future OCR / parser / Neo4j calls.

Without a tool layer, timeout handling and conflicts become scattered across scripts.

## Current implementation

Core code:

```text
src/paper_agent/tools/orchestrator.py
```

Concepts:

- `ToolSpec`: registered tool metadata and handler.
- `ToolCall`: one requested call.
- `ToolResult`: isolated result for success, error, timeout or unknown tool.
- `ResourceAccess`: `read` or `write`.
- `ToolOrchestrator`: batches compatible calls and serializes conflicts.
- `ToolCall.depends_on`: declares a dependency DAG, so dependent calls wait for
  prerequisite results.

Conflict policy:

- read/read on the same resource can run in the same batch;
- write/read or write/write on the same resource are split into different batches;
- unknown tools return `UNKNOWN_TOOL` instead of crashing the whole run;
- timeout returns `TIMEOUT` for that call instead of crashing the whole run.
- transient failures can be retried with `ToolSpec.max_retries` and exponential
  backoff via `retry_backoff_s`;
- each result records `attempts` and `elapsed_ms` for a lightweight execution trace;
- unresolved dependencies become structured `ERROR` results.

## Demo

```bash
python scripts/demo_tool_orchestrator.py
```

Expected behavior:

- retrieval and graph query can share a batch because they read different resources;
- session memory read and write are split into different batches because write conflicts
  with read on the same resource;
- slow tool returns `timeout` instead of blocking the whole demo.

## Paper QA plan demo

M4-2 adds a real paper-QA pre-answer plan:

```text
ContextGuard -> retrieve_chunks + query_graph_concepts + recall_summary_memory
             -> update_session
```

The plan stops before LLM answer generation, so it is cheap to test and does not require
GPU models. It proves that the project has a real tool-calling path for retrieval,
graph lookup and memory update.

The normal `scripts/ask.py` path also routes chunk retrieval through this boundary, so
the end-to-end demo exercises timeout, retry and structured-result handling.

Run:

```bash
python scripts/demo_paper_qa_tool_plan.py
```

Expected behavior:

- ambiguous query is constrained to the current session paper;
- retrieval receives the resolved `paper_id`;
- summary memory recall retrieves relevant compressed episodic memory;
- session memory is updated through a write tool;
- each tool returns a structured `ToolResult`.

## Resume framing

This gives the project a clear tool-calling layer without pretending to be a full
distributed system:

> Built a lightweight tool orchestration layer for a paper-reading agent, supporting
> tool registration, per-tool timeout isolation, resource-aware conflict scheduling and
> structured execution results.

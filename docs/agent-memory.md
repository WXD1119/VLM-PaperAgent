# Agent Memory Design

This project separates agent memory from the paper knowledge graph.

The paper knowledge graph should model paper content only: papers, sections, chunks,
figures, tables, equations, concepts and eventually claims extracted from the papers
themselves. User preferences, conversation history, generated answers and runtime task
state belong to agent memory or artifacts, not to the paper graph.

## Three-layer agent memory

### 1. Short-term session memory

Purpose: keep the current run coherent.

Examples:

- current workspace ID;
- current paper ID;
- last query;
- last answer artifact path;
- active task.

Implementation:

- `paper_agent.memory.SessionState`
- `paper_agent.memory.SessionMemoryStore`
- default path: `artifacts/memory/session.json`

### 2. Episodic memory

Purpose: remember what happened without treating it as paper knowledge.

Examples:

- a user added `LLaVA.pdf`;
- an answer was generated and saved as an artifact;
- a retrieval evaluation was run;
- the user clarified that paper graph must not store user profile data.

Implementation:

- `paper_agent.memory.Episode`
- `paper_agent.memory.EpisodicMemoryStore`
- default path: `artifacts/memory/episodes.jsonl`

### 3. Long-term user profile memory

Purpose: store stable user and environment preferences.

Examples:

- preferred language;
- default workspace ID;
- server project path;
- Conda environment name;
- preferred command style.

Implementation:

- `paper_agent.memory.UserProfile`
- `paper_agent.memory.UserProfileStore`
- default path: `artifacts/memory/user_profile.json`

## Artifact memory

Answer bundles, evaluation outputs and parsed paper artifacts are reproducible project
artifacts. They are not graph nodes by default.

Examples:

- `artifacts/answers/*.answer.json`
- `artifacts/evals/*.json`
- `artifacts/papers/{paper_id}/paper.json`
- `artifacts/papers/{paper_id}/chunks.json`

Useful command:

```bash
python scripts/list_answers.py --answers artifacts/answers

python scripts/list_answers.py --answers artifacts/answers --contains llava --abstained false
```

## Promotion policy

An answer can be considered for long-term memory only if it is reusable and evidence
grounded. Even then, the policy should ask the user instead of silently persisting it.

Current policy:

- abstained answers are rejected;
- answers with no claims are rejected;
- answers whose semantic judge report is not fully supported are rejected;
- otherwise the agent may ask the user whether to archive or save it.

This policy is implemented in `paper_agent.memory.PromotionPolicy`.

## M2 memory policy

M2 adds a general `MemoryPolicy` that decides where a memory candidate belongs. It is a
router, not a writer: it explains the action and target, but does not persist data.

Vocabulary:

- `candidate`: information that might be remembered.
- `policy`: the rule set that decides what to do.
- `action`: what should happen, such as `write`, `keep`, `ask_user`, `ignore` or
  `route_outside_memory`.
- `target`: where the information belongs, such as `session`, `episodic`, `profile`,
  `artifact`, `paper_kg` or `none`.

Default routing:

| Candidate kind | Action | Target | Meaning |
| --- | --- | --- | --- |
| `task_state` | `write` | `session` | Current working context. |
| `project_event` | `write` | `episodic` | Append-only project event. |
| `user_preference` with explicit request | `write` | `profile` | Stable user preference. |
| inferred `user_preference` | `ask_user` | `profile` | Confirm before persistence. |
| `answer_artifact` | `keep` | `artifact` | Keep generated answer as a file artifact. |
| `paper_content` | `route_outside_memory` | `paper_kg` | Paper content belongs in Paper KG. |
| `casual_chat` | `ignore` | `none` | Do not pollute memory. |

Useful command:

```bash
python scripts/memory_decide.py \
  --kind user_preference \
  --content "Prefer Chinese explanations" \
  --explicit-user-request

python scripts/memory_decide.py \
  --kind paper_content \
  --content "LLaVA connects visual features to language embeddings" \
  --json
```

Policy-gated write command:

```bash
python scripts/memory_apply.py \
  --kind project_event \
  --content "Ran LLaVA memory policy test" \
  --metadata-json '{"event_type":"test_event"}'

python scripts/memory_apply.py \
  --kind paper_content \
  --content "LLaVA architecture evidence"
```

Expected behavior:

- `project_event` writes an episode and prints `written: true`.
- `paper_content` prints `target: paper_kg` and `written: false`, because Paper KG is
  outside agent memory.
- inferred `user_preference` prints `action: ask_user` and `written: false`, unless the
  command includes `--explicit-user-request` and a profile key.

Existing write commands now use the same gate:

- `scripts/memory_log_event.py` routes events as `project_event` before writing
  episodic memory.
- `scripts/memory_profile.py --set` routes preferences as explicit `user_preference`
  before updating profile memory.

## M1 workflow integration

The first workflow integration step makes memory useful during normal CLI work:

- `scripts/ask.py` updates short-term session memory after saving an answer bundle:
  `last_query`, `last_answer_path`, `current_paper_id` and runtime metadata.
- `scripts/add_paper_to_workspace.py` appends a `paper_added_to_workspace` episode after
  a successful workspace commit.
- `scripts/list_answers.py` lists answer artifacts and supports `--contains`,
  `--abstained true|false` and `--latest`.
- `scripts/query_workspace.py` queries a workspace effective graph directly, so users no
  longer need inline Python to inspect their personal graph branch.
- `scripts/memory_show.py` prints the current session state, user profile and recent
  episodes in one place.
- `scripts/memory_show.py --json` prints the same state as machine-readable JSON for a
  future API or website.
- `scripts/ask.py --log-episode` can append an `answer_generated` episode after writing
  an answer artifact. Without this flag, `ask.py` updates session memory only.
- `scripts/ask.py` prints a user-facing `Memory` section after answer generation. It
  explains that the answer is kept as an artifact, not written to Paper KG, and whether
  the user should be asked before long-term archiving.
- `scripts/ask.py --show-memory-policy` prints the developer-facing routing details for
  debugging and tests.
- `/ask` API responses expose the same user-facing summary as a structured `memory`
  object, so a future website can render memory status without parsing CLI text.

M1 acceptance checks:

```bash
python scripts/list_answers.py --answers artifacts/answers --contains llava --latest 5

python scripts/query_workspace.py --workspace artifacts/graph_workspaces/ws_wxd_demo --papers

python scripts/memory_show.py --recent 5

python scripts/memory_show.py --recent 5 --json
```

After one `ask.py` run, `artifacts/memory/session.json` should contain the latest query
and answer artifact path. After one `ask.py --log-episode` run, `episodes.jsonl` should
contain an `answer_generated` event.

For answer generation, the default memory output is intentionally user-facing:

```text
# Memory
Status: answer saved as artifact; not written to Paper KG.
Recommendation: ask user before long-term archiving.
```

For developer debugging, `ask.py --show-memory-policy` prints:

```text
# Memory policy
artifact_status: kept
paper_kg_status: not_written
promotion_verdict: ask_user|reject
```

The API returns the same default summary as structured JSON:

```json
{
  "memory": {
    "status": "answer saved as artifact; not written to Paper KG",
    "artifact_status": "saved",
    "paper_kg_written": false,
    "recommendation": "ask user before long-term archiving",
    "requires_user_confirmation": true
  }
}
```

## Relationship to the paper graph

```text
Agent Memory            Paper Knowledge Graph
-------------           ---------------------
session state           Paper
episodes                Section
user profile            Chunk
answer artifacts        Figure / Table / Equation
evaluation artifacts    Concept
                        PaperClaim (future)
```

The graph can be used by the agent as an external knowledge store, but it is not the
agent's memory system.

## M3 context guard

M3 starts using memory to reduce ambiguous follow-up errors and multi-turn topic drift.
The first implementation is deterministic and testable:

- if the user provides an explicit `paper_id`, that always wins;
- if the query is an ambiguous follow-up such as "this method" or "这个方法", and session
  memory has a `current_paper_id`, retrieval is constrained to that paper;
- if the query is ambiguous and no session paper exists, the guard asks for
  clarification instead of searching the whole corpus;
- if the query is a clear new question, the guard does not force it into the previous
  session paper.

Useful command:

```bash
python scripts/memory_context.py \
  --query "这个方法怎么连接视觉和语言？"
```

In `scripts/ask.py`, the context guard is enabled by default. Use
`--no-context-guard` for ablation or debugging.

The `/ask` API also exposes the guard decision as `context`:

```json
{
  "context": {
    "action": "constrain",
    "resolved_paper_id": "paper_dc8bace378a282ea",
    "needs_clarification": false,
    "warnings": [
      "ambiguous follow-up constrained to current_paper_id from session memory"
    ]
  }
}
```

If a query needs clarification, `/ask` returns an abstained answer with the
clarification question instead of running retrieval over the whole corpus.

M3 also adds a context-guard evaluation set:

```bash
python scripts/evaluate_context_guard.py \
  --golden evals/context_guard.seed.json \
  --output artifacts/evals/context_guard.seed.json
```

Metrics:

- `Accuracy`: exact decision match across action, resolved paper and clarification flag.
- `Constraint recall`: among ambiguous follow-ups that should be constrained, how many
  are constrained to the right paper.
- `Clarification recall`: among missing-context follow-ups, how many ask clarification.
- `Wrong constraint rate`: among clear new questions, how often the guard incorrectly
  binds the query to an old session paper.

## M5 episodic summary memory

M5 adds the second memory layer that many agents use: sliding-window context plus
compressed summary recall.

Current implementation:

- `SlidingWindowMemory`: keeps the latest `N` turns active and returns older turns as
  overflow.
- `HeuristicConversationSummarizer`: deterministic offline summarizer for overflow
  turns. It extracts summary text, paper IDs, key entities, decisions and unresolved
  questions.
- `SummaryMemoryStore`: append-only JSONL store for compressed summaries.
- `InMemorySummaryVectorStore`: lightweight vector recall over summaries using a local
  hashing embedder.

This is intentionally separate from the paper chunk vector store:

```text
Paper vector DB              Agent summary memory
---------------              --------------------
paper chunks                 conversation / task summaries
evidence retrieval           history recall
Paper KG compatible          not written to Paper KG
```

Useful commands:

```bash
python scripts/memory_summarize.py \
  --episodes artifacts/memory/episodes.jsonl \
  --summaries artifacts/memory/summaries.jsonl \
  --cursor artifacts/memory/summary_cursor.json \
  --window-size 6

python scripts/memory_recall.py \
  --query "LLaVA context recall" \
  --summaries artifacts/memory/summaries.jsonl \
  --top-k 5
```

The current vector recall uses dependency-free hashing embeddings for reproducible tests.
A production deployment can swap it for BGE-M3 + Chroma using the same
`ConversationSummary.embedding_text` field and a separate collection such as
`agent_memory_summaries_bge_m3`.

Compression trigger:

- manual trigger: run `scripts/memory_summarize.py`;
- the script reads `summary_cursor.json` and only considers episodes after
  `last_summarized_event_id`;
- if pending events are fewer than or equal to `--window-size`, no summary is written;
- if pending events exceed `--window-size`, older overflow events are summarized and the
  newest `--window-size` events remain active;
- after writing a summary, the cursor is advanced to the last event included in that
  summary, preventing repeated compression of the same events.

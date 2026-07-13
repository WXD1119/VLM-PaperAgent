# M5: Episodic Summary Memory

M5 implements the agent's second memory layer.

The goal is not to store more raw chat. The goal is to keep recent context small and
recover useful older context when needed.

## Pipeline

```text
recent turns
  -> SlidingWindowMemory
  -> overflow turns
  -> ConversationSummary
  -> SummaryMemoryStore
  -> Summary vector recall
```

## Why separate from Paper KG

Paper KG should contain paper facts and evidence structure only.

Summary memory contains user-agent interaction history:

- project decisions;
- unresolved user questions;
- current implementation milestones;
- paper IDs discussed during the session;
- operational context that helps future turns.

Therefore summary memory must not be merged into Paper KG.

## Current storage

Default path:

```text
artifacts/memory/summaries.jsonl
artifacts/memory/summary_cursor.json
```

Each record is a `ConversationSummary` with:

- `summary`;
- `source_turn_ids`;
- `key_entities`;
- `paper_ids`;
- `decisions`;
- `unresolved_questions`;
- metadata.

## Current vector recall

The first implementation uses `HashingTextEmbedder` and `InMemorySummaryVectorStore`.

This is intentionally lightweight:

- no GPU required;
- no network required;
- deterministic tests;
- same schema can later be backed by BGE-M3 + Chroma.

Future production collection name:

```text
agent_memory_summaries_bge_m3
```

This must stay separate from the paper chunk collection.

## Compression trigger

The first implementation uses explicit/manual compression:

```bash
python scripts/memory_summarize.py \
  --episodes artifacts/memory/episodes.jsonl \
  --summaries artifacts/memory/summaries.jsonl \
  --cursor artifacts/memory/summary_cursor.json \
  --window-size 6
```

Compression happens only when pending events after the cursor exceed `window-size`.

The cursor records:

- `last_summarized_event_id`;
- `last_summary_id`;
- `updated_at`.

This prevents duplicate summaries when the command is run repeatedly.

Future automatic triggers can call the same logic after every `N` answer turns or before
context construction in `/ask`.

## Display time

Memory records store timestamps in UTC, for example:

```text
2026-07-13T07:20:29.063889+00:00
```

Display commands convert this to local time for readability. For China Standard Time,
`memory_show.py` displays both:

```text
2026-07-13 15:20:29 CST (2026-07-13T07:20:29.063889+00:00)
```

Storage stays UTC; presentation can use `--timezone`.

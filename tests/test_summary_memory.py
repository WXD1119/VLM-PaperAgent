import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("summary memory tests require pydantic v2", allow_module_level=True)

from paper_agent.memory import (
    ConversationTurn,
    EpisodicMemoryStore,
    HeuristicConversationSummarizer,
    SlidingWindowMemory,
    SummaryMemoryStore,
    SummaryCursorStore,
    build_summary_vector_store,
    compress_episode_memory,
    summarize_new_episodes,
)
from paper_agent.memory.time import format_local_time


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_SUMMARIZE_SCRIPT = REPO_ROOT / "scripts" / "memory_summarize.py"


def test_sliding_window_splits_recent_turns_from_overflow():
    turns = [ConversationTurn(role="user", content=f"turn {index}") for index in range(5)]

    result = SlidingWindowMemory(max_turns=2).split(turns)

    assert [turn.content for turn in result.overflow] == ["turn 0", "turn 1", "turn 2"]
    assert [turn.content for turn in result.active_window] == ["turn 3", "turn 4"]


def test_format_local_time_displays_utc_timestamp_in_shanghai_time():
    rendered = format_local_time("2026-07-13T07:20:29.063889+00:00")

    assert rendered.startswith("2026-07-13 15:20:29")


def test_heuristic_summarizer_extracts_paper_ids_entities_and_questions():
    turns = [
        ConversationTurn(
            role="event",
            content="Decided to keep Paper KG separate from Agent Memory for paper_abc123.",
            metadata={"paper_id": "paper_abc123"},
        ),
        ConversationTurn(role="user", content="How should LLaVA memory recall work?"),
    ]

    summary = HeuristicConversationSummarizer().summarize(turns)

    assert summary is not None
    assert summary.paper_ids == ["paper_abc123"]
    assert "LLaVA" in summary.key_entities
    assert summary.decisions
    assert summary.unresolved_questions == ["How should LLaVA memory recall work?"]


def test_summary_store_and_vector_recall_are_separate_from_paper_chunks(tmp_path):
    summary = HeuristicConversationSummarizer().summarize(
        [
            ConversationTurn(
                role="event",
                content="Decided LLaVA uses summary memory for context recall.",
                metadata={"paper_id": "paper_llava"},
            )
        ]
    )
    store = SummaryMemoryStore(tmp_path / "summaries.jsonl")
    store.append(summary)

    loaded = store.list()
    vector_store = build_summary_vector_store(loaded)
    hits = vector_store.search("LLaVA context recall", top_k=1)

    assert loaded[0].summary_id == summary.summary_id
    assert hits[0].summary.summary_id == summary.summary_id
    assert hits[0].score > 0


def test_memory_summarize_cli_writes_overflow_summary(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    summaries = tmp_path / "summaries.jsonl"
    cursor = tmp_path / "summary_cursor.json"
    episode_store = EpisodicMemoryStore(episodes)
    for index in range(4):
        episode_store.log(
            "event",
            f"Decided memory step {index} for paper_llava",
            {"paper_id": "paper_llava"},
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(MEMORY_SUMMARIZE_SCRIPT),
            "--episodes",
            str(episodes),
            "--summaries",
            str(summaries),
            "--cursor",
            str(cursor),
            "--window-size",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    loaded = SummaryMemoryStore(summaries).list()
    assert "overflow: 2" in completed.stdout
    assert len(loaded) == 1
    assert loaded[0].paper_ids == ["paper_llava"]
    assert SummaryCursorStore(cursor).load().last_summarized_event_id == loaded[0].source_turn_ids[-1]


def test_summary_cursor_prevents_repeated_compression(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    summaries = tmp_path / "summaries.jsonl"
    cursor = tmp_path / "summary_cursor.json"
    episode_store = EpisodicMemoryStore(episodes)
    for index in range(4):
        episode_store.log(
            "event",
            f"Decided cursor step {index} for paper_llava",
            {"paper_id": "paper_llava"},
        )

    first = subprocess.run(
        [
            sys.executable,
            str(MEMORY_SUMMARIZE_SCRIPT),
            "--episodes",
            str(episodes),
            "--summaries",
            str(summaries),
            "--cursor",
            str(cursor),
            "--window-size",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    second = subprocess.run(
        [
            sys.executable,
            str(MEMORY_SUMMARIZE_SCRIPT),
            "--episodes",
            str(episodes),
            "--summaries",
            str(summaries),
            "--cursor",
            str(cursor),
            "--window-size",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "written: true" in first.stdout
    assert "summary: none" in second.stdout
    assert "written: false" in second.stdout
    assert len(SummaryMemoryStore(summaries).list()) == 1


def test_summarize_new_episodes_waits_until_pending_exceeds_window(tmp_path):
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")
    episodes = [
        episode_store.log("event", "one"),
        episode_store.log("event", "two"),
    ]

    result = summarize_new_episodes(episodes, window_size=2)

    assert result.pending_events == 2
    assert result.active_window_events == 2
    assert result.overflow_events == 0
    assert result.summary is None


def test_compress_episode_memory_writes_once_after_window_overflow(tmp_path):
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")
    summary_store = SummaryMemoryStore(tmp_path / "summaries.jsonl")
    cursor_store = SummaryCursorStore(tmp_path / "summary_cursor.json")
    for index in range(4):
        episode_store.log("event", f"paper_abc123 decision {index}", {"paper_id": "paper_abc123"})

    result = compress_episode_memory(
        episode_store,
        summary_store,
        cursor_store,
        window_size=2,
    )

    assert result.written
    assert result.overflow_events == 2
    assert len(summary_store.list()) == 1
    assert cursor_store.load().last_summarized_event_id == result.summary.source_turn_ids[-1]

    again = compress_episode_memory(
        episode_store,
        summary_store,
        cursor_store,
        window_size=2,
    )
    assert not again.written
    assert len(summary_store.list()) == 1

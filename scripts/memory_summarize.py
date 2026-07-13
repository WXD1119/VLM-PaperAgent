import argparse

from paper_agent.memory import (
    EpisodicMemoryStore,
    SummaryCursorStore,
    SummaryMemoryStore,
    summarize_new_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress old episodic memory into summary memory using a sliding window"
    )
    parser.add_argument("--episodes", default="artifacts/memory/episodes.jsonl")
    parser.add_argument("--summaries", default="artifacts/memory/summaries.jsonl")
    parser.add_argument("--cursor", default="artifacts/memory/summary_cursor.json")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the summary without writing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = EpisodicMemoryStore(args.episodes).list()
    cursor_store = SummaryCursorStore(args.cursor)
    result = summarize_new_episodes(
        episodes,
        window_size=args.window_size,
        cursor=cursor_store.load(),
        metadata={
            "source": args.episodes,
        },
    )
    summary = result.summary

    print(f"episodes: {result.total_events}")
    print(f"pending: {result.pending_events}")
    print(f"active_window: {result.active_window_events}")
    print(f"overflow: {result.overflow_events}")
    print(f"cursor_before: {result.cursor.last_summarized_event_id or 'none'}")
    if summary is None:
        print("summary: none")
        print("written: false")
        return

    if not args.dry_run:
        SummaryMemoryStore(args.summaries).append(summary)
        cursor_store.update(
            event_id=summary.source_turn_ids[-1],
            summary_id=summary.summary_id,
        )
    print(f"summary_id: {summary.summary_id}")
    print(f"written: {str(not args.dry_run).lower()}")
    print(f"cursor_after: {summary.source_turn_ids[-1] if not args.dry_run else 'unchanged'}")
    print(f"paper_ids: {summary.paper_ids}")
    print(f"key_entities: {summary.key_entities}")
    print(f"decisions: {len(summary.decisions)}")
    print(f"unresolved_questions: {len(summary.unresolved_questions)}")
    print(f"summary: {summary.summary}")


if __name__ == "__main__":
    main()

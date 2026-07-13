import argparse
import json

from paper_agent.memory.time import format_local_time
from paper_agent.memory import (
    EpisodicMemoryStore,
    SessionMemoryStore,
    SummaryMemoryStore,
    UserProfileStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show agent memory state")
    parser.add_argument("--session", default="artifacts/memory/session.json")
    parser.add_argument("--profile", default="artifacts/memory/user_profile.json")
    parser.add_argument("--episodes", default="artifacts/memory/episodes.jsonl")
    parser.add_argument("--summaries", default="artifacts/memory/summaries.jsonl")
    parser.add_argument("--recent", type=int, default=5)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    session = SessionMemoryStore(args.session).load()
    profile = UserProfileStore(args.profile).load()
    episodes = EpisodicMemoryStore(args.episodes).list(limit=args.recent)
    summaries = SummaryMemoryStore(args.summaries).list(limit=args.recent)
    payload = {
        "session": session.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "recent_episodes": [episode.model_dump(mode="json") for episode in episodes],
        "recent_summaries": [summary.model_dump(mode="json") for summary in summaries],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("# Session Memory")
    print(session.model_dump_json(indent=2))

    print("\n# User Profile Memory")
    print(profile.model_dump_json(indent=2))

    print(f"\n# Recent Episodes last={args.recent}")
    if not episodes:
        print("(none)")
    else:
        for episode in episodes:
            print(
                f"- {format_local_time(episode.timestamp, timezone=args.timezone)} "
                f"({episode.timestamp}) {episode.event_type} {episode.event_id}"
            )
            print(f"  {episode.summary}")
            if episode.payload:
                print(f"  payload={episode.payload}")

    print(f"\n# Recent Summaries last={args.recent}")
    if not summaries:
        print("(none)")
        return
    for summary in summaries:
        print(
            f"- {format_local_time(summary.created_at, timezone=args.timezone)} "
            f"({summary.created_at}) {summary.summary_id}"
        )
        print(f"  paper_ids={summary.paper_ids}")
        print(f"  key_entities={summary.key_entities}")
        print(f"  {summary.summary}")


if __name__ == "__main__":
    main()

import argparse
import json

from paper_agent.memory import EpisodicMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Append an event to episodic agent memory")
    parser.add_argument("--memory", default="artifacts/memory/episodes.jsonl")
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--payload-json",
        default="{}",
        help="Optional JSON object with structured event details.",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload_json)
    if not isinstance(payload, dict):
        raise SystemExit("--payload-json must decode to a JSON object")
    episode = EpisodicMemoryStore(args.memory).log(args.event_type, args.summary, payload)
    print(f"event_id: {episode.event_id}")
    print(f"timestamp: {episode.timestamp}")
    print(f"event_type: {episode.event_type}")
    print(f"summary: {episode.summary}")


if __name__ == "__main__":
    main()

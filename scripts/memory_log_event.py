import argparse
import json

from memory_apply import apply_memory_decision
from paper_agent.memory import MemoryCandidate, MemoryCandidateKind, MemoryPolicy


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
    candidate = MemoryCandidate(
        kind=MemoryCandidateKind.PROJECT_EVENT,
        content=args.summary,
        explicit_user_request=True,
        metadata={"event_type": args.event_type, **{str(k): str(v) for k, v in payload.items()}},
    )
    decision = MemoryPolicy().decide(candidate)
    result = apply_memory_decision(
        candidate,
        decision,
        session_path="artifacts/memory/session.json",
        profile_path="artifacts/memory/user_profile.json",
        episodes_path=args.memory,
    )
    print(f"action: {decision.action.value}")
    print(f"target: {decision.target.value}")
    print(f"written: {str(result['written']).lower()}")
    if result.get("record_id"):
        print(f"event_id: {result['record_id']}")
    print(f"event_type: {args.event_type}")
    print(f"summary: {args.summary}")
    print("reasons:")
    for reason in decision.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

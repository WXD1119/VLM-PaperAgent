import argparse
import json

from paper_agent.memory import (
    EpisodicMemoryStore,
    MemoryAction,
    MemoryCandidate,
    MemoryCandidateKind,
    MemoryPolicy,
    MemoryTarget,
    SessionMemoryStore,
    UserProfileStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply MemoryPolicy and write only policy-approved memory records"
    )
    parser.add_argument("--kind", required=True, choices=[kind.value for kind in MemoryCandidateKind])
    parser.add_argument("--content", required=True)
    parser.add_argument("--explicit-user-request", action="store_true")
    parser.add_argument("--session", default="artifacts/memory/session.json")
    parser.add_argument("--profile", default="artifacts/memory/user_profile.json")
    parser.add_argument("--episodes", default="artifacts/memory/episodes.jsonl")
    parser.add_argument(
        "--metadata-json",
        default="{}",
        help="Optional JSON object. For profile writes, include key/value.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json)
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must decode to a JSON object")
    candidate = MemoryCandidate(
        kind=MemoryCandidateKind(args.kind),
        content=args.content,
        explicit_user_request=args.explicit_user_request,
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
    decision = MemoryPolicy().decide(candidate)
    write_result = apply_memory_decision(
        candidate,
        decision,
        session_path=args.session,
        profile_path=args.profile,
        episodes_path=args.episodes,
    )
    payload = {
        "decision": decision.model_dump(mode="json"),
        "written": write_result["written"],
        "write_target": write_result["write_target"],
        "record_id": write_result.get("record_id"),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"action: {decision.action.value}")
    print(f"target: {decision.target.value}")
    print(f"written: {str(write_result['written']).lower()}")
    if write_result.get("record_id"):
        print(f"record_id: {write_result['record_id']}")
    print("reasons:")
    for reason in decision.reasons:
        print(f"- {reason}")


def apply_memory_decision(
    candidate: MemoryCandidate,
    decision,
    *,
    session_path: str,
    profile_path: str,
    episodes_path: str,
) -> dict[str, str | bool | None]:
    if decision.action != MemoryAction.WRITE:
        return {"written": False, "write_target": None}
    if decision.target == MemoryTarget.SESSION:
        SessionMemoryStore(session_path).update(
            current_workspace_id=candidate.metadata.get("current_workspace_id"),
            current_paper_id=candidate.metadata.get("current_paper_id"),
            last_query=candidate.metadata.get("last_query"),
            last_answer_path=candidate.metadata.get("last_answer_path"),
            active_task=candidate.metadata.get("active_task") or candidate.content,
            metadata={"memory_apply_content": candidate.content, **candidate.metadata},
        )
        return {"written": True, "write_target": decision.target.value}
    if decision.target == MemoryTarget.EPISODIC:
        episode = EpisodicMemoryStore(episodes_path).log(
            candidate.metadata.get("event_type", "memory_event"),
            candidate.content,
            dict(candidate.metadata),
        )
        return {
            "written": True,
            "write_target": decision.target.value,
            "record_id": episode.event_id,
        }
    if decision.target == MemoryTarget.PROFILE:
        key = candidate.metadata.get("key")
        value = candidate.metadata.get("value", candidate.content)
        if not key:
            raise SystemExit("profile writes require --metadata-json with a 'key' field")
        UserProfileStore(profile_path).set(key, value)
        return {"written": True, "write_target": decision.target.value}
    return {"written": False, "write_target": None}


if __name__ == "__main__":
    main()

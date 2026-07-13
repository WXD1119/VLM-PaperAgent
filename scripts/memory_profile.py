import argparse

from memory_apply import apply_memory_decision
from paper_agent.memory import MemoryCandidate, MemoryCandidateKind, MemoryPolicy, UserProfileStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or update long-term user profile memory")
    parser.add_argument("--memory", default="artifacts/memory/user_profile.json")
    parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"))
    args = parser.parse_args()

    store = UserProfileStore(args.memory)
    if args.set:
        key, value = args.set
        candidate = MemoryCandidate(
            kind=MemoryCandidateKind.USER_PREFERENCE,
            content=value,
            explicit_user_request=True,
            metadata={"key": key, "value": value},
        )
        decision = MemoryPolicy().decide(candidate)
        result = apply_memory_decision(
            candidate,
            decision,
            session_path="artifacts/memory/session.json",
            profile_path=args.memory,
            episodes_path="artifacts/memory/episodes.jsonl",
        )
        if not result["written"]:
            print(f"not updated: {key}")
            print(f"action: {decision.action.value}")
            print(f"target: {decision.target.value}")
            for reason in decision.reasons:
                print(f"- {reason}")
            raise SystemExit(1)
        profile = store.load()
        print(f"updated: {key}")
    else:
        profile = store.load()
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

import argparse
import json

from paper_agent.memory import MemoryCandidate, MemoryCandidateKind, MemoryPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain how a memory candidate should be routed")
    parser.add_argument("--kind", required=True, choices=[kind.value for kind in MemoryCandidateKind])
    parser.add_argument("--content", required=True)
    parser.add_argument(
        "--explicit-user-request",
        action="store_true",
        help="Mark that the user explicitly asked to remember this information.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    decision = MemoryPolicy().decide(
        MemoryCandidate(
            kind=MemoryCandidateKind(args.kind),
            content=args.content,
            explicit_user_request=args.explicit_user_request,
        )
    )
    if args.json:
        print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    print(f"action: {decision.action.value}")
    print(f"target: {decision.target.value}")
    print(f"requires_user_confirmation: {decision.requires_user_confirmation}")
    print("reasons:")
    for reason in decision.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

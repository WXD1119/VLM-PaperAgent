import argparse

from paper_agent.memory import list_answer_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="List immutable answer artifacts")
    parser.add_argument("--answers", default="artifacts/answers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--latest", type=int, default=None, help="Show the latest N artifacts")
    parser.add_argument("--contains", default=None, help="Case-insensitive query/path substring filter")
    parser.add_argument(
        "--abstained",
        choices=("true", "false"),
        default=None,
        help="Filter by abstention status",
    )
    args = parser.parse_args()

    summaries = list_answer_artifacts(args.answers)
    if args.contains:
        needle = args.contains.lower()
        summaries = [
            item
            for item in summaries
            if needle in item.query.lower() or needle in item.path.lower()
        ]
    if args.abstained is not None:
        expected = args.abstained == "true"
        summaries = [item for item in summaries if item.abstained == expected]
    limit = args.latest if args.latest is not None else args.limit
    if limit is not None:
        summaries = summaries[-limit:]
    if not summaries:
        print(f"no answer artifacts found under {args.answers}")
        return

    print("abstained\tclaims\tevidence\tmodel\tpath\tquery")
    for item in summaries:
        print(
            f"{str(item.abstained).lower()}\t{item.claim_count}\t{item.evidence_count}\t"
            f"{item.generator_model or '-'}\t{item.path}\t{item.query}"
        )


if __name__ == "__main__":
    main()

import argparse

from paper_agent.memory import list_answer_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="List immutable answer artifacts")
    parser.add_argument("--answers", default="artifacts/answers")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summaries = list_answer_artifacts(args.answers)
    if args.limit is not None:
        summaries = summaries[-args.limit :]
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

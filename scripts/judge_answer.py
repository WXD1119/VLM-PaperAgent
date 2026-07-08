import argparse
from pathlib import Path

from paper_agent.agents import SemanticCitationJudge
from paper_agent.domain import AnswerBundle
from paper_agent.llm import GlmStructuredClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge a saved answer after the generator exits")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-memory-gib", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    bundle = AnswerBundle.model_validate_json(args.input.read_text(encoding="utf-8"))
    client = GlmStructuredClient(
        args.model,
        max_memory_gib=args.max_memory_gib,
        max_new_tokens=args.max_new_tokens,
    )
    report = SemanticCitationJudge(client).evaluate(bundle.answer, bundle.evidence_pack)

    if bundle.answer.abstained:
        print("Semantic citation validation: SKIPPED (answer abstained)")
    else:
        print("Semantic citation validation:")
        for item in report.assessments:
            print(
                f"- claim={item.claim_index} verdict={item.verdict.value} "
                f"evidence={item.evidence_ids}: {item.reasoning_summary}"
            )
        print(f"All claims supported: {report.all_supported}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Judge report: {args.output}")


if __name__ == "__main__":
    main()

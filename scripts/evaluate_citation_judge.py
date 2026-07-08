import argparse
from pathlib import Path

from paper_agent.domain import SemanticCitationReport
from paper_agent.evaluation.citation import CitationGoldenSet, evaluate_citation_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GLM judge against human labels")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    golden = CitationGoldenSet.model_validate_json(args.golden.read_text(encoding="utf-8"))
    predictions = {
        path.stem.removesuffix(".glm-judge"): SemanticCitationReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in args.predictions.glob("*.glm-judge.json")
    }
    result = evaluate_citation_judge(golden, predictions)
    print(f"cases: {result.case_count}")
    print(f"claims: {result.claim_count}")
    print(f"Claim accuracy: {result.claim_accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Supported precision: {result.supported_precision:.4f}")
    print(f"Supported recall: {result.supported_recall:.4f}")
    print(f"Abstention accuracy: {result.abstention_accuracy:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()

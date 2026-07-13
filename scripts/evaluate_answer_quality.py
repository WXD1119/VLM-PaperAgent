import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle, SemanticCitationReport
from paper_agent.evaluation import evaluate_answer_quality, evaluate_answer_quality_gate


def _case_id_from_answer(path: Path) -> str:
    return path.name.removesuffix(".answer.json")


def _case_id_from_judge(path: Path) -> str:
    return path.name.removesuffix(".glm-judge.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate answer faithfulness and hallucination-risk metrics"
    )
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument(
        "--judges",
        type=Path,
        help="Directory containing optional *.glm-judge.json semantic citation reports.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-semantic-coverage", type=float)
    parser.add_argument("--min-semantic-support", type=float)
    parser.add_argument("--max-unsupported-claim-rate", type=float)
    parser.add_argument("--min-fully-supported-answer-rate", type=float)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with status 1 when any configured quality gate fails.",
    )
    args = parser.parse_args()

    answer_paths = sorted(args.answers.glob("*.answer.json"))
    answers = {
        _case_id_from_answer(path): AnswerBundle.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        for path in answer_paths
    }
    reports = {}
    if args.judges and args.judges.exists():
        reports = {
            _case_id_from_judge(path): SemanticCitationReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in sorted(args.judges.glob("*.glm-judge.json"))
        }

    result = evaluate_answer_quality(answers, reports)
    print(f"answers: {result.answer_count}")
    print(f"claims: {result.claim_count}")
    print(f"Citation pass rate: {result.citation_pass_rate:.4f}")
    print(f"Abstention rate: {result.abstention_rate:.4f}")
    print(f"Semantic coverage rate: {result.semantic_coverage_rate:.4f}")
    print(f"Semantic support rate: {result.semantic_support_rate:.4f}")
    print(f"Unsupported claim rate: {result.unsupported_claim_rate:.4f}")
    print(f"Fully supported answer rate: {result.fully_supported_answer_rate:.4f}")
    for case in result.cases:
        print(
            f"{case.case_id}: claims={case.claim_count} "
            f"citation_valid={case.citation_valid} abstained={case.abstained} "
            f"semantic_support={case.semantic_support_rate:.3f} "
            f"unsupported={case.unsupported_claim_rate:.3f} "
            f"fully_supported={case.fully_supported}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"output: {args.output}")

    gate = evaluate_answer_quality_gate(
        result,
        min_semantic_coverage=args.min_semantic_coverage,
        min_semantic_support=args.min_semantic_support,
        max_unsupported_claim_rate=args.max_unsupported_claim_rate,
        min_fully_supported_answer_rate=args.min_fully_supported_answer_rate,
    )
    if any(
        value is not None
        for value in (
            args.min_semantic_coverage,
            args.min_semantic_support,
            args.max_unsupported_claim_rate,
            args.min_fully_supported_answer_rate,
        )
    ):
        print(f"Quality gate: {'PASS' if gate.passed else 'FAIL'}")
        for failure in gate.failures:
            print(
                f"- {failure.metric}: actual={failure.actual:.4f} "
                f"expected={failure.expected}; {failure.reason}"
            )
        if gate.risky_cases:
            print("Risky cases:")
            for case_id in gate.risky_cases:
                print(f"- {case_id}")
        if args.fail_on_gate and not gate.passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

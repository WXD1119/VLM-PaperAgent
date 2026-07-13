import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from paper_agent.evaluation import ContextGuardCase, evaluate_context_guard


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory-aware context guard decisions")
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw_cases = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[ContextGuardCase]).validate_python(raw_cases)
    result = evaluate_context_guard(cases)

    print(f"cases: {result.case_count}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Constraint recall: {result.constraint_recall:.4f}")
    print(f"Clarification recall: {result.clarification_recall:.4f}")
    print(f"Wrong constraint rate: {result.wrong_constraint_rate:.4f}")
    for case in result.cases:
        print(
            f"{case.case_id}: expected={case.expected_action.value} "
            f"predicted={case.predicted_action.value} "
            f"expected_paper={case.expected_resolved_paper_id or '*'} "
            f"predicted_paper={case.predicted_resolved_paper_id or '*'} "
            f"correct={case.correct}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()

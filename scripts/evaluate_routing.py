"""评测 Query Router 并输出结构化结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_agent.evaluation.routing import RoutingGoldenCase, evaluate_routing


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper query routing")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_cases = json.loads(args.golden.read_text(encoding="utf-8"))
    report = evaluate_routing([RoutingGoldenCase.model_validate(item) for item in raw_cases])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"cases: {report.case_count}")
    print(f"Intent accuracy: {report.intent_accuracy:.4f}")
    print(f"Evidence type Macro-F1: {report.evidence_type_macro_f1:.4f}")
    print(f"Graph routing accuracy: {report.graph_routing_accuracy:.4f}")
    print(f"Clarification accuracy: {report.clarification_accuracy:.4f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def find_completed_experiment(registry: list[dict[str, Any]], experiment_id: str) -> dict[str, Any] | None:
    for item in registry:
        if item.get("experiment_id") == experiment_id and item.get("status") == "completed":
            return item
    return None


def find_best_completed_retrieval_ablation(registry: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [
        item
        for item in registry
        if item.get("status") == "completed" and item.get("category") == "retrieval_ablation"
    ]
    if not completed:
        return None
    return max(completed, key=lambda item: int(item.get("dataset", {}).get("cases") or 0))


def render_report(
    *,
    registry: list[dict[str, Any]] | None = None,
    answer_quality: dict[str, Any] | None = None,
    context_guard: dict[str, Any] | None = None,
    title: str = "VLM PaperAgent Experiment Report",
) -> str:
    registry = registry or []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {title}",
        "",
        f"Generated at: {now}",
        "",
        "## Executive Summary",
        "",
        "- Evidence-grounded paper QA pipeline with citation validation and semantic citation judging.",
        "- Retrieval stack compares BM25, dense retrieval, hybrid RRF and cross-encoder reranking.",
        "- Agent memory is separated from the Paper KG: user/session state stays in memory; paper content stays in graph.",
        "- Quality gates make hallucination-risk evaluation reproducible instead of anecdotal.",
        "",
    ]

    retrieval = find_best_completed_retrieval_ablation(registry)
    if retrieval:
        lines.extend(render_retrieval_section(retrieval))
    else:
        lines.extend([
            "## Retrieval Ablation",
            "",
            "No completed retrieval ablation was found in the experiment registry.",
            "",
        ])

    if answer_quality:
        lines.extend(render_answer_quality_section(answer_quality))
    else:
        lines.extend([
            "## Answer Quality and Hallucination Risk",
            "",
            "No answer quality JSON was provided.",
            "",
        ])

    if context_guard:
        lines.extend(render_context_guard_section(context_guard))
    else:
        lines.extend([
            "## Context Guard",
            "",
            "No context guard evaluation JSON was provided.",
            "",
        ])

    lines.extend(render_engineering_sections(registry))
    lines.extend(render_resume_section(answer_quality=answer_quality, retrieval=retrieval))
    return "\n".join(lines).rstrip() + "\n"


def render_retrieval_section(experiment: dict[str, Any]) -> list[str]:
    metrics = experiment.get("metrics", {})
    lines = [
        "## Retrieval Ablation",
        "",
        f"Dataset: `{experiment.get('dataset', {}).get('name', '-')}`; "
        f"cases: {experiment.get('dataset', {}).get('cases', '-')}.",
        "",
        "| Variant | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in metrics.items():
        lines.append(
            f"| {variant} | {fmt(row.get('hit_at_1'))} | "
            f"{fmt(row.get('hit_at_5'))} | {fmt(row.get('recall_at_1'))} | "
            f"{fmt(row.get('recall_at_5'))} | {fmt(row.get('mrr'))} | "
            f"{fmt(row.get('ndcg_at_5'))} |"
        )
    lines.extend([
        "",
        f"Conclusion: {experiment.get('conclusion', '-')}",
        "",
    ])
    return lines


def render_answer_quality_section(answer_quality: dict[str, Any]) -> list[str]:
    lines = [
        "## Answer Quality and Hallucination Risk",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Answers | {answer_quality.get('answer_count', '-')} |",
        f"| Claims | {answer_quality.get('claim_count', '-')} |",
        f"| Citation pass rate | {pct(answer_quality.get('citation_pass_rate'))} |",
        f"| Semantic coverage rate | {pct(answer_quality.get('semantic_coverage_rate'))} |",
        f"| Semantic support rate | {pct(answer_quality.get('semantic_support_rate'))} |",
        f"| Unsupported claim rate | {pct(answer_quality.get('unsupported_claim_rate'))} |",
        f"| Fully supported answer rate | {pct(answer_quality.get('fully_supported_answer_rate'))} |",
        "",
    ]
    if answer_quality.get("semantic_coverage_rate") == 1.0 and answer_quality.get("unsupported_claim_rate") == 0:
        lines.append(
            "Quality gate result: PASS. All generated claims were semantically judged, "
            "and no unsupported claim was found in the evaluated answer set."
        )
    else:
        lines.append(
            "Quality gate result: inspect thresholds and risky cases before presenting "
            "these results as final."
        )
    lines.append("")
    return lines


def render_context_guard_section(context_guard: dict[str, Any]) -> list[str]:
    return [
        "## Context Guard",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases | {context_guard.get('case_count', '-')} |",
        f"| Accuracy | {pct(context_guard.get('accuracy'))} |",
        f"| Constraint recall | {pct(context_guard.get('constraint_recall'))} |",
        f"| Clarification recall | {pct(context_guard.get('clarification_recall'))} |",
        f"| Wrong constraint rate | {pct(context_guard.get('wrong_constraint_rate'))} |",
        "",
    ]


def render_engineering_sections(registry: list[dict[str, Any]]) -> list[str]:
    completed = [item for item in registry if item.get("status") == "completed"]
    lines = [
        "## Engineering Validation",
        "",
        "| Experiment | Category | Conclusion |",
        "|---|---|---|",
    ]
    for item in completed:
        if item.get("category") in {
            "engineering_comparison",
            "answer_citation_evaluation",
            "retrieval_baseline",
        }:
            lines.append(
                f"| `{item.get('experiment_id')}` | {item.get('category')} | "
                f"{item.get('conclusion', '-')} |"
            )
    lines.append("")
    return lines


def render_resume_section(
    *,
    answer_quality: dict[str, Any] | None,
    retrieval: dict[str, Any] | None,
) -> list[str]:
    retrieval_text = ""
    if retrieval:
        best = retrieval.get("metrics", {}).get("RRF_BGE_reranker_v2_m3", {})
        hit_text = ""
        if best.get("hit_at_1") is not None:
            hit_text = f"Hit@1={fmt(best.get('hit_at_1'))}, "
        retrieval_text = (
            f" Best reranked retrieval reached {hit_text}Recall@1={fmt(best.get('recall_at_1'))}, "
            f"Recall@5={fmt(best.get('recall_at_5'))}, nDCG@5={fmt(best.get('ndcg_at_5'))}."
        )
    answer_text = ""
    if answer_quality:
        answer_text = (
            f" Semantic judge covered {answer_quality.get('claim_count', '-')} claims "
            f"across {answer_quality.get('answer_count', '-')} answers with "
            f"unsupported-claim rate {pct(answer_quality.get('unsupported_claim_rate'))}."
        )
    return [
        "## Resume-Ready Summary",
        "",
        (
            "- Built an evidence-grounded paper reading agent with MinerU parsing, "
            "hybrid retrieval, reranking, citation-valid answer generation, independent "
            "semantic judging, graph workspaces and memory-aware context control."
            + retrieval_text
            + answer_text
        ),
        "",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a markdown experiment report")
    parser.add_argument("--registry", type=Path, default=Path("evals/experiment_registry.json"))
    parser.add_argument("--answer-quality", type=Path, default=Path("artifacts/evals/answer_quality.json"))
    parser.add_argument("--context-guard", type=Path, default=Path("artifacts/evals/context_guard.seed.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/experiment_report.md"))
    parser.add_argument("--title", default="VLM PaperAgent Experiment Report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry) or []
    answer_quality = load_json(args.answer_quality)
    context_guard = load_json(args.context_guard)
    report = render_report(
        registry=registry,
        answer_quality=answer_quality,
        context_guard=context_guard,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

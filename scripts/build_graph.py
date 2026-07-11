import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle, Paper
from paper_agent.graph import GraphBuilder, GraphValidator, write_graph_jsonl
from paper_agent.retrieval import load_chunk_bundles


def load_papers(root: str | Path) -> list[Paper]:
    root_path = Path(root)
    paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("paper.json"))
    if not paths:
        raise FileNotFoundError(f"no paper.json found below {root_path}")
    return [Paper.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]


def load_answer_bundles(root: str | Path) -> list[AnswerBundle]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*.answer.json"))
    return [
        AnswerBundle.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight paper knowledge graph")
    parser.add_argument("--papers", default="artifacts/papers")
    parser.add_argument("--chunks", default=None, help="Defaults to --papers")
    parser.add_argument(
        "--answers",
        default="artifacts/answers",
        help="Legacy answer bundle input, used only with --include-answers.",
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help=(
            "Legacy traceability mode: include Query/Answer/Claim nodes. "
            "The default Paper KG excludes user/agent QA records."
        ),
    )
    parser.add_argument("--output", default="artifacts/graph")
    parser.add_argument("--allow-empty-answers", action="store_true")
    args = parser.parse_args()

    papers = load_papers(args.papers)
    chunks = load_chunk_bundles(args.chunks or args.papers)
    answers = load_answer_bundles(args.answers) if args.include_answers else []
    if args.include_answers and not answers and not args.allow_empty_answers:
        raise FileNotFoundError(
            f"no *.answer.json found below {args.answers}; pass --allow-empty-answers "
            "or omit --include-answers to build only the paper knowledge graph"
        )

    builder = GraphBuilder()
    builder.add_papers(papers)
    builder.add_chunks(chunks)
    if args.include_answers:
        builder.add_answers(answers)
    graph = builder.build()

    report = GraphValidator().validate(graph)
    print(f"papers: {len(papers)}")
    print(f"chunk_bundles: {len(chunks)}")
    print(f"answer_bundles: {len(answers)}")
    print(f"nodes: {report.node_count}")
    print(f"edges: {report.edge_count}")
    print(f"valid: {report.valid}")
    if not report.valid:
        for error in report.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    nodes_path, edges_path = write_graph_jsonl(graph, args.output)
    print(f"nodes_output: {nodes_path}")
    print(f"edges_output: {edges_path}")


if __name__ == "__main__":
    main()

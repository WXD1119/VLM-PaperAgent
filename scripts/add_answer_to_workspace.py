import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle
from paper_agent.graph import (
    GraphValidator,
    LocalGraphWorkspaceStore,
    build_answer_fragment,
    delta_from_fragment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a saved answer bundle to a graph workspace")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--answer", required=True, help="Path to *.answer.json")
    parser.add_argument("--author", required=True)
    parser.add_argument("--message", default=None)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    answer = AnswerBundle.model_validate_json(Path(args.answer).read_text(encoding="utf-8"))
    fragment = build_answer_fragment(answer)
    store = LocalGraphWorkspaceStore(args.workspace)
    delta = delta_from_fragment(store.load_effective_graph(), fragment)

    if not delta.added_nodes and not delta.added_edges:
        print(f"no new graph records for answer query: {answer.evidence_pack.query}")
        return

    preview = store.preview_delta(delta)
    report = GraphValidator().validate(preview)
    if not report.valid and not args.allow_invalid:
        print("answer delta would make effective graph invalid:")
        for error in report.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    commit = store.commit_delta(
        delta,
        author_id=args.author,
        message=args.message or f"add answer for query: {answer.evidence_pack.query}",
    )
    print(f"query: {answer.evidence_pack.query}")
    print(f"commit_id: {commit.commit_id}")
    print(f"valid_after_commit: {report.valid}")
    print(f"added_nodes: {len(delta.added_nodes)}")
    print(f"added_edges: {len(delta.added_edges)}")


if __name__ == "__main__":
    main()

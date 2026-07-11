import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle
from paper_agent.graph import (
    LocalGraphWorkspaceStore,
    promote_answer_to_workspace,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper: promote a saved answer bundle to long-term graph memory. "
            "Prefer scripts/promote_answer_to_workspace.py for new workflows."
        )
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--answer", required=True, help="Path to *.answer.json")
    parser.add_argument("--author", required=True)
    parser.add_argument("--message", default=None)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    store = LocalGraphWorkspaceStore(args.workspace)
    answer = AnswerBundle.model_validate_json(Path(args.answer).read_text(encoding="utf-8"))
    print(
        "warning: add_answer_to_workspace.py is a legacy compatibility wrapper; "
        "new product flows should keep generated answers in artifact/agent memory."
    )
    result = promote_answer_to_workspace(
        answer,
        store,
        author_id=args.author,
        message=args.message or f"add answer for query: {answer.evidence_pack.query}",
        allow_invalid=args.allow_invalid,
    )

    if not result.validation.valid and not args.allow_invalid:
        print("answer promotion rejected: delta would make effective graph invalid:")
        for error in result.validation.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    if not result.changed:
        print(f"no new graph records for answer query: {result.query}")
        return

    print(f"query: {result.query}")
    print(f"commit_id: {result.commit.commit_id if result.commit else '(dry-run)'}")
    print(f"valid_after_commit: {result.validation.valid}")
    print(f"added_nodes: {len(result.delta.added_nodes)}")
    print(f"added_edges: {len(result.delta.added_edges)}")


if __name__ == "__main__":
    main()

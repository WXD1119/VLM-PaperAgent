import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle
from paper_agent.graph import LocalGraphWorkspaceStore, promote_answer_to_workspace


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY: promote a saved AnswerBundle into a graph workspace. "
            "New product flows should keep generated answers in artifact/agent memory "
            "instead of mixing user/agent QA records into the Paper KG."
        )
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--answer", required=True, help="Path to *.answer.json")
    parser.add_argument("--author", required=True)
    parser.add_argument("--message", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()

    answer = AnswerBundle.model_validate_json(Path(args.answer).read_text(encoding="utf-8"))
    store = LocalGraphWorkspaceStore(args.workspace)
    print(
        "warning: answer-to-workspace promotion is legacy traceability mode; "
        "Paper KG should normally contain paper content only."
    )
    result = promote_answer_to_workspace(
        answer,
        store,
        author_id=args.author,
        message=args.message,
        allow_invalid=args.allow_invalid,
        dry_run=args.dry_run,
    )

    if not result.validation.valid and not args.allow_invalid:
        print("answer promotion rejected: delta would make effective graph invalid")
        for error in result.validation.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"query: {result.query}")
    print(f"dry_run: {args.dry_run}")
    print(f"valid_after_promotion: {result.validation.valid}")
    print(f"added_nodes: {len(result.delta.added_nodes)}")
    print(f"added_edges: {len(result.delta.added_edges)}")
    if not result.changed:
        print("status: no-op; answer records are already visible in the workspace")
    elif args.dry_run:
        print("status: preview only; no workspace commit was written")
    else:
        print("status: promoted")
        print(f"commit_id: {result.commit.commit_id if result.commit else '(none)'}")


if __name__ == "__main__":
    main()

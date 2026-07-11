import argparse
from pathlib import Path

from paper_agent.graph import (
    GraphDelta,
    GraphEdge,
    GraphNode,
    GraphValidator,
    LocalGraphWorkspaceStore,
)


def read_nodes(path: str | None) -> list[GraphNode]:
    if not path:
        return []
    return [
        GraphNode.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_edges(path: str | None) -> list[GraphEdge]:
    if not path:
        return []
    return [
        GraphEdge.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_ids(path: str | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit a graph delta into a workspace")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--nodes", default=None, help="JSONL file of GraphNode records to add")
    parser.add_argument("--edges", default=None, help="JSONL file of GraphEdge records to add")
    parser.add_argument("--removed-nodes", default=None, help="Line-delimited node IDs to hide")
    parser.add_argument("--removed-edges", default=None, help="Line-delimited edge IDs to hide")
    parser.add_argument("--author", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Commit even if the resulting effective graph fails validation",
    )
    args = parser.parse_args()

    delta = GraphDelta(
        added_nodes=read_nodes(args.nodes),
        added_edges=read_edges(args.edges),
        removed_node_ids=read_ids(args.removed_nodes),
        removed_edge_ids=read_ids(args.removed_edges),
    )
    if not any(
        [delta.added_nodes, delta.added_edges, delta.removed_node_ids, delta.removed_edge_ids]
    ):
        parser.error("delta is empty; provide --nodes, --edges, --removed-nodes or --removed-edges")

    store = LocalGraphWorkspaceStore(args.workspace)
    preview = store.preview_delta(delta)
    report = GraphValidator().validate(preview)
    if not report.valid and not args.allow_invalid:
        print("delta would make effective graph invalid:")
        for error in report.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    commit = store.commit_delta(delta, author_id=args.author, message=args.message)
    print(f"commit_id: {commit.commit_id}")
    print(f"parent_commit_id: {commit.parent_commit_id}")
    print(f"message: {commit.message}")
    print(f"valid_after_commit: {report.valid}")
    print(f"stats: {commit.stats}")


if __name__ == "__main__":
    main()

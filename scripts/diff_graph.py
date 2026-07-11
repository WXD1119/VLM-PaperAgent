import argparse

from paper_agent.graph import LocalGraphWorkspaceStore, diff_graphs, read_graph_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff a base graph and workspace effective graph")
    parser.add_argument("--base", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--show-ids", action="store_true")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    base_graph = read_graph_jsonl(args.base)
    workspace_graph = LocalGraphWorkspaceStore(args.workspace).load_effective_graph()
    diff = diff_graphs(base_graph, workspace_graph)

    print(f"added_nodes: {len(diff.added_node_ids)}")
    print(f"added_edges: {len(diff.added_edge_ids)}")
    print(f"removed_nodes: {len(diff.removed_node_ids)}")
    print(f"removed_edges: {len(diff.removed_edge_ids)}")
    print(f"added_node_types: {diff.added_node_types}")
    print(f"added_edge_types: {diff.added_edge_types}")
    print(f"removed_node_types: {diff.removed_node_types}")
    print(f"removed_edge_types: {diff.removed_edge_types}")

    if args.show_ids:
        _print_ids("added_node_ids", diff.added_node_ids, args.limit)
        _print_ids("added_edge_ids", diff.added_edge_ids, args.limit)
        _print_ids("removed_node_ids", diff.removed_node_ids, args.limit)
        _print_ids("removed_edge_ids", diff.removed_edge_ids, args.limit)


def _print_ids(label: str, values: list[str], limit: int) -> None:
    print(f"{label}:")
    for value in values[:limit]:
        print(f"- {value}")
    if len(values) > limit:
        print(f"- ... {len(values) - limit} more")


if __name__ == "__main__":
    main()

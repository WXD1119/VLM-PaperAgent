import argparse
from collections import Counter

from paper_agent.graph import GraphValidator, LocalGraphWorkspaceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a graph workspace effective graph")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    store = LocalGraphWorkspaceStore(args.workspace)
    workspace = store.read_workspace()
    commits = store.commit_chain()
    graph = store.load_effective_graph()
    report = GraphValidator().validate(graph)
    node_types = Counter(node.node_type.value for node in graph.nodes)
    edge_types = Counter(edge.edge_type.value for edge in graph.edges)

    print(f"workspace_id: {workspace.workspace_id}")
    print(f"owner_id: {workspace.owner_id}")
    print(f"name: {workspace.name}")
    print(f"base_graph_path: {workspace.base_graph_path}")
    print(f"head_commit_id: {workspace.head_commit_id}")
    print(f"commits: {len(commits)}")
    print(f"valid: {report.valid}")
    print(f"nodes: {report.node_count}")
    print(f"edges: {report.edge_count}")
    print(f"node_types: {dict(sorted(node_types.items()))}")
    print(f"edge_types: {dict(sorted(edge_types.items()))}")
    if args.show_errors and report.errors:
        print("errors:")
        for error in report.errors:
            print(f"- {error}")
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

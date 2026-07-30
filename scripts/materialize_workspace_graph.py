import argparse
from pathlib import Path

from paper_agent.graph import LocalGraphWorkspaceStore, GraphValidator, write_graph_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a workspace effective graph into portable JSONL files"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = LocalGraphWorkspaceStore(args.workspace).load_effective_graph()
    report = GraphValidator().validate(graph)
    if not report.valid:
        raise SystemExit("workspace graph is invalid: " + "; ".join(report.errors))
    nodes_path, edges_path = write_graph_jsonl(graph, args.output)
    print(f"workspace: {args.workspace}")
    print(f"valid: {report.valid}")
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")
    print(f"nodes_path: {nodes_path}")
    print(f"edges_path: {edges_path}")


if __name__ == "__main__":
    main()

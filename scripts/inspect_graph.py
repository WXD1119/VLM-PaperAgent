import argparse
from collections import Counter

from paper_agent.graph import GraphValidator, read_graph_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a lightweight evidence graph")
    parser.add_argument("--graph", default="artifacts/graph")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    graph = read_graph_jsonl(args.graph)
    report = GraphValidator().validate(graph)
    node_types = Counter(node.node_type.value for node in graph.nodes)
    edge_types = Counter(edge.edge_type.value for edge in graph.edges)
    paper_nodes = [node for node in graph.nodes if node.node_type.value == "Paper"]

    print(f"valid: {report.valid}")
    print(f"nodes: {report.node_count}")
    print(f"edges: {report.edge_count}")
    print(f"node_types: {dict(sorted(node_types.items()))}")
    print(f"edge_types: {dict(sorted(edge_types.items()))}")
    print("papers:")
    for node in paper_nodes:
        print(f"- {node.properties.get('paper_id')}: {node.label}")
    if args.show_errors and report.errors:
        print("errors:")
        for error in report.errors:
            print(f"- {error}")

    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

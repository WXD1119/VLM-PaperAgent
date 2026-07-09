from pathlib import Path

from paper_agent.graph.model import GraphDocument, GraphEdge, GraphNode


def write_graph_jsonl(graph: GraphDocument, output: str | Path) -> tuple[Path, Path]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    nodes_path = output_path / "nodes.jsonl"
    edges_path = output_path / "edges.jsonl"
    nodes_path.write_text(
        "\n".join(node.model_dump_json() for node in graph.nodes) + "\n",
        encoding="utf-8",
    )
    edges_path.write_text(
        "\n".join(edge.model_dump_json() for edge in graph.edges) + "\n",
        encoding="utf-8",
    )
    return nodes_path, edges_path


def read_graph_jsonl(root: str | Path) -> GraphDocument:
    root_path = Path(root)
    nodes = [
        GraphNode.model_validate_json(line)
        for line in (root_path / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    edges = [
        GraphEdge.model_validate_json(line)
        for line in (root_path / "edges.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return GraphDocument(nodes=nodes, edges=edges)

from collections import defaultdict
from collections.abc import Iterable

from paper_agent.graph.model import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType


class GraphQuery:
    """Small in-memory query helper for the lightweight evidence graph."""

    def __init__(self, graph: GraphDocument) -> None:
        self.graph = graph
        self.nodes_by_id = {node.node_id: node for node in graph.nodes}
        self.out_edges: dict[str, list[GraphEdge]] = defaultdict(list)
        self.in_edges: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in graph.edges:
            self.out_edges[edge.source_id].append(edge)
            self.in_edges[edge.target_id].append(edge)

    def papers(self) -> list[GraphNode]:
        return sorted(
            (node for node in self.graph.nodes if node.node_type == NodeType.PAPER),
            key=lambda node: str(node.properties.get("paper_id", node.node_id)),
        )

    def chunks_for_paper(self, paper_id: str, limit: int = 20) -> list[GraphNode]:
        paper_node_id = f"paper:{paper_id}"
        chunks: list[GraphNode] = []
        for edge in self.out_edges.get(paper_node_id, []):
            if edge.edge_type == EdgeType.HAS_CHUNK:
                node = self.nodes_by_id[edge.target_id]
                chunks.append(node)
        return sorted(
            chunks,
            key=lambda node: (
                min(node.properties.get("pages", [10**9])),
                node.properties.get("chunk_id", node.node_id),
            ),
        )[:limit]

    def claim_supports(self, limit: int = 20) -> list[tuple[GraphNode, list[GraphNode]]]:
        claims = sorted(
            (node for node in self.graph.nodes if node.node_type == NodeType.CLAIM),
            key=lambda node: (node.properties.get("claim_index", 0), node.node_id),
        )
        rows: list[tuple[GraphNode, list[GraphNode]]] = []
        for claim in claims:
            chunks = [
                self.nodes_by_id[edge.target_id]
                for edge in self.out_edges.get(claim.node_id, [])
                if edge.edge_type == EdgeType.SUPPORTED_BY
            ]
            rows.append((claim, chunks))
        return rows[:limit]

    def search_nodes(
        self,
        text: str,
        node_types: Iterable[NodeType] | None = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        needle = text.lower()
        allowed = set(node_types) if node_types else None
        matches: list[GraphNode] = []
        for node in self.graph.nodes:
            if allowed and node.node_type not in allowed:
                continue
            haystack = " ".join(
                [
                    node.label,
                    str(node.properties.get("paper_id", "")),
                    str(node.properties.get("title", "")),
                    str(node.properties.get("query", "")),
                    str(node.properties.get("text", "")),
                    str(node.properties.get("content_preview", "")),
                ]
            ).lower()
            if needle in haystack:
                matches.append(node)
        return sorted(matches, key=lambda node: (node.node_type.value, node.node_id))[:limit]

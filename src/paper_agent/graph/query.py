from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from paper_agent.graph.concepts import normalize_concept
from paper_agent.graph.model import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType


@dataclass
class ConceptNeighborhood:
    concept: GraphNode | None
    candidates: list[GraphNode] = field(default_factory=list)
    papers: list[GraphNode] = field(default_factory=list)
    sections: list[GraphNode] = field(default_factory=list)
    claims: list[tuple[GraphNode, list[GraphNode]]] = field(default_factory=list)
    chunks: list[GraphNode] = field(default_factory=list)
    related_concepts: list[GraphNode] = field(default_factory=list)


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

    def concept_candidates(self, keyword: str, limit: int = 10) -> list[GraphNode]:
        """Return possible concept nodes for a keyword without forcing disambiguation."""

        normalized = normalize_concept(keyword)
        exact_id = f"concept:{normalized}"
        exact = self.nodes_by_id.get(exact_id)
        candidates: dict[str, GraphNode] = {}
        if exact is not None:
            candidates[exact.node_id] = exact
        needle = keyword.lower()
        normalized_needle = normalized.replace("-", " ")
        for node in self.graph.nodes:
            if node.node_type != NodeType.CONCEPT:
                continue
            aliases = node.properties.get("aliases", [])
            fields = [
                node.label,
                str(node.properties.get("canonical_name", "")),
                str(node.properties.get("normalized", "")).replace("-", " "),
                *[str(alias) for alias in aliases],
            ]
            haystack = " | ".join(fields).lower()
            if needle in haystack or normalized_needle in haystack:
                candidates[node.node_id] = node
        return _sort_nodes(candidates.values())[:limit]

    def concept_neighborhood(self, keyword: str, limit: int = 10) -> ConceptNeighborhood:
        candidates = self.concept_candidates(keyword, limit=limit)
        normalized = normalize_concept(keyword)
        exact = self.nodes_by_id.get(f"concept:{normalized}")
        concept = exact if exact is not None else (candidates[0] if len(candidates) == 1 else None)
        if concept is None:
            return ConceptNeighborhood(concept=None, candidates=candidates)

        mention_sources = [
            self.nodes_by_id[edge.source_id]
            for edge in self.in_edges.get(concept.node_id, [])
            if edge.edge_type == EdgeType.MENTIONS and edge.source_id in self.nodes_by_id
        ]
        chunks = _unique_nodes(
            [node for node in mention_sources if node.node_type == NodeType.CHUNK]
        )
        direct_claims = _unique_nodes(
            [node for node in mention_sources if node.node_type == NodeType.CLAIM]
        )

        claims_with_evidence: list[tuple[GraphNode, list[GraphNode]]] = []
        claim_ids = {claim.node_id for claim in direct_claims}
        for claim in direct_claims:
            evidence_chunks = [
                self.nodes_by_id[edge.target_id]
                for edge in self.out_edges.get(claim.node_id, [])
                if edge.edge_type == EdgeType.SUPPORTED_BY and edge.target_id in self.nodes_by_id
            ]
            chunks.extend(evidence_chunks)
            claims_with_evidence.append((claim, _unique_nodes(evidence_chunks)))

        chunks = _unique_nodes(chunks)
        paper_ids = {f"paper:{chunk.properties.get('paper_id')}" for chunk in chunks}
        papers = _unique_nodes(
            [self.nodes_by_id[paper_id] for paper_id in paper_ids if paper_id in self.nodes_by_id]
        )
        section_ids = {
            edge.source_id
            for chunk in chunks
            for edge in self.in_edges.get(chunk.node_id, [])
            if edge.edge_type == EdgeType.CONTAINS
        }
        sections = _unique_nodes(
            [
                self.nodes_by_id[section_id]
                for section_id in section_ids
                if section_id in self.nodes_by_id
            ]
        )

        related_concept_ids: set[str] = set()
        for source in [*chunks, *direct_claims]:
            for edge in self.out_edges.get(source.node_id, []):
                if (
                    edge.edge_type == EdgeType.MENTIONS
                    and edge.target_id != concept.node_id
                    and edge.target_id in self.nodes_by_id
                ):
                    related_concept_ids.add(edge.target_id)
        related_concepts = _unique_nodes(
            [self.nodes_by_id[node_id] for node_id in related_concept_ids]
        )

        return ConceptNeighborhood(
            concept=concept,
            candidates=candidates,
            papers=_sort_nodes(papers)[:limit],
            sections=_sort_nodes(sections)[:limit],
            claims=sorted(
                claims_with_evidence,
                key=lambda row: (row[0].properties.get("claim_index", 0), row[0].node_id),
            )[:limit],
            chunks=sorted(
                chunks,
                key=lambda node: (
                    str(node.properties.get("paper_id", "")),
                    min(node.properties.get("pages", [10**9])),
                    node.properties.get("chunk_id", node.node_id),
                ),
            )[:limit],
            related_concepts=_sort_nodes(related_concepts)[:limit],
        )


def _unique_nodes(nodes: Iterable[GraphNode]) -> list[GraphNode]:
    unique: dict[str, GraphNode] = {}
    for node in nodes:
        unique[node.node_id] = node
    return list(unique.values())


def _sort_nodes(nodes: Iterable[GraphNode]) -> list[GraphNode]:
    return sorted(nodes, key=lambda node: (node.label.lower(), node.node_id))

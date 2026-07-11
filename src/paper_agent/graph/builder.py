import hashlib
from collections.abc import Iterable

from paper_agent.domain import AnswerBundle, ChunkBundle, Paper, RetrievalChunk
from paper_agent.graph.concepts import ConceptMention, extract_concepts
from paper_agent.graph.model import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType


def stable_id(*parts: object, prefix: str = "") -> str:
    raw = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


class GraphBuilder:
    """Build a lightweight evidence graph from parsed papers, chunks and answers."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

    def add_papers(self, papers: Iterable[Paper]) -> None:
        for paper in papers:
            self._add_node(
                GraphNode(
                    node_id=paper_node_id(paper.paper_id),
                    node_type=NodeType.PAPER,
                    label=paper.title,
                    properties={
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "source_path": paper.source_path,
                        "sha256": paper.sha256,
                        "element_count": len(paper.elements),
                    },
                )
            )

    def add_chunks(self, bundles: Iterable[ChunkBundle]) -> None:
        for bundle in bundles:
            paper_id = paper_node_id(bundle.paper_id)
            for chunk in bundle.children:
                self._add_sections_for_chunk(chunk)
                chunk_id = chunk_node_id(chunk.chunk_id)
                section_id = section_node_id(chunk.paper_id, chunk.section_path)
                self._add_node(
                    GraphNode(
                        node_id=chunk_id,
                        node_type=NodeType.CHUNK,
                        label=chunk.chunk_id,
                        properties={
                            "chunk_id": chunk.chunk_id,
                            "parent_chunk_id": chunk.parent_chunk_id,
                            "paper_id": chunk.paper_id,
                            "kind": chunk.kind.value,
                            "pages": chunk.pages,
                            "section_path": chunk.section_path,
                            "element_ids": chunk.element_ids,
                            "content_preview": preview(chunk.content),
                        },
                    )
                )
                self._add_edge(
                    paper_id,
                    chunk_id,
                    EdgeType.HAS_CHUNK,
                    properties={"paper_id": chunk.paper_id},
                )
                self._add_edge(
                    section_id,
                    chunk_id,
                    EdgeType.CONTAINS,
                    properties={"paper_id": chunk.paper_id},
                )
                self._add_concept_mentions(
                    source_id=chunk_id,
                    text=" ".join([*chunk.section_path, chunk.content]),
                    source_kind="chunk",
                    extra_properties={
                        "paper_id": chunk.paper_id,
                        "chunk_id": chunk.chunk_id,
                        "section_path": chunk.section_path,
                        "pages": chunk.pages,
                    },
                )

    def add_answers(self, bundles: Iterable[AnswerBundle]) -> None:
        for bundle in bundles:
            query_id = query_node_id(bundle.evidence_pack.query)
            answer_id = answer_node_id(
                bundle.evidence_pack.query,
                bundle.answer.answer,
                bundle.generator_model,
            )
            self._add_node(
                GraphNode(
                    node_id=query_id,
                    node_type=NodeType.QUERY,
                    label=bundle.evidence_pack.query,
                    properties={"query": bundle.evidence_pack.query},
                )
            )
            self._add_node(
                GraphNode(
                    node_id=answer_id,
                    node_type=NodeType.ANSWER,
                    label=preview(bundle.answer.answer, 100),
                    properties={
                        "query": bundle.evidence_pack.query,
                        "answer": bundle.answer.answer,
                        "abstained": bundle.answer.abstained,
                        "abstention_reason": bundle.answer.abstention_reason,
                        "generator_model": bundle.generator_model,
                    },
                )
            )
            self._add_edge(query_id, answer_id, EdgeType.ANSWERED_WITH)
            evidence_by_id = {
                item.evidence_id: item for item in bundle.evidence_pack.items
            }
            for claim_index, claim in enumerate(bundle.answer.claims, start=1):
                claim_id = claim_node_id(answer_id, claim_index)
                self._add_node(
                    GraphNode(
                        node_id=claim_id,
                        node_type=NodeType.CLAIM,
                        label=preview(claim.text, 120),
                        properties={
                            "claim_index": claim_index,
                            "text": claim.text,
                            "evidence_ids": claim.evidence_ids,
                        },
                    )
                )
                self._add_edge(
                    answer_id,
                    claim_id,
                    EdgeType.HAS_CLAIM,
                    properties={"claim_index": claim_index},
                )
                for evidence_id in claim.evidence_ids:
                    item = evidence_by_id[evidence_id]
                    self._add_edge(
                        claim_id,
                        chunk_node_id(item.chunk_id),
                        EdgeType.SUPPORTED_BY,
                        properties={
                            "evidence_id": evidence_id,
                            "paper_id": item.paper_id,
                            "pages": item.pages,
                        },
                    )
                self._add_concept_mentions(
                    source_id=claim_id,
                    text=claim.text,
                    source_kind="claim",
                    extra_properties={"claim_index": claim_index},
                )

    def build(self) -> GraphDocument:
        nodes = sorted(self._nodes.values(), key=lambda item: item.node_id)
        edges = sorted(self._edges.values(), key=lambda item: item.edge_id)
        return GraphDocument(nodes=nodes, edges=edges)

    def _add_sections_for_chunk(self, chunk: RetrievalChunk) -> None:
        paper_id = paper_node_id(chunk.paper_id)
        parent_id = paper_id
        for depth in range(1, len(chunk.section_path) + 1):
            path = chunk.section_path[:depth]
            section_id = section_node_id(chunk.paper_id, path)
            self._add_node(
                GraphNode(
                    node_id=section_id,
                    node_type=NodeType.SECTION,
                    label=path[-1],
                    properties={
                        "paper_id": chunk.paper_id,
                        "section_path": path,
                        "depth": depth,
                    },
                )
            )
            self._add_edge(
                parent_id,
                section_id,
                EdgeType.HAS_SECTION,
                properties={"paper_id": chunk.paper_id, "depth": depth},
            )
            parent_id = section_id
        if not chunk.section_path:
            root_id = section_node_id(chunk.paper_id, [])
            self._add_node(
                GraphNode(
                    node_id=root_id,
                    node_type=NodeType.SECTION,
                    label="(unknown section)",
                    properties={"paper_id": chunk.paper_id, "section_path": [], "depth": 0},
                )
            )
            self._add_edge(paper_id, root_id, EdgeType.HAS_SECTION)

    def _add_node(self, node: GraphNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing and existing != node:
            raise ValueError(f"conflicting graph node: {node.node_id}")
        self._nodes[node.node_id] = node

    def _add_concept_mentions(
        self,
        *,
        source_id: str,
        text: str,
        source_kind: str,
        extra_properties: dict,
    ) -> None:
        for mention in extract_concepts(text, source="rule"):
            self._add_concept_node(mention)
            self._add_edge(
                source_id,
                concept_node_id(mention.normalized),
                EdgeType.MENTIONS,
                properties={
                    "concept": mention.normalized,
                    "mention": mention.label,
                    "source_kind": source_kind,
                    "context_preview": preview(text),
                    **extra_properties,
                },
            )

    def _add_concept_node(self, mention: ConceptMention) -> None:
        node_id = concept_node_id(mention.normalized)
        existing = self._nodes.get(node_id)
        if existing is not None:
            aliases = set(existing.properties.get("aliases", []))
            aliases.add(mention.label)
            existing.properties["aliases"] = sorted(aliases, key=str.lower)
            return
        self._add_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.CONCEPT,
                label=mention.label,
                properties={
                    "canonical_name": mention.label,
                    "name": mention.label,
                    "normalized": mention.normalized,
                    "aliases": [mention.label],
                    "disambiguation_key": mention.normalized,
                    "resolver": "deterministic-v1",
                    "source": mention.source,
                },
            )
        )

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        properties: dict | None = None,
    ) -> None:
        edge_id = edge_node_id(source_id, edge_type.value, target_id, properties or {})
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties or {},
        )
        self._edges[edge_id] = edge


def preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return "." * limit
    return compact[: limit - 3].rstrip() + "..."


def paper_node_id(paper_id: str) -> str:
    return f"paper:{paper_id}"


def section_node_id(paper_id: str, section_path: list[str]) -> str:
    return "section:" + stable_id(paper_id, *section_path)


def chunk_node_id(chunk_id: str) -> str:
    return f"chunk:{chunk_id}"


def query_node_id(query: str) -> str:
    return "query:" + stable_id(query)


def answer_node_id(query: str, answer: str, generator_model: str) -> str:
    return "answer:" + stable_id(query, answer, generator_model)


def claim_node_id(answer_id: str, claim_index: int) -> str:
    return f"claim:{stable_id(answer_id, claim_index)}"


def concept_node_id(normalized: str) -> str:
    return f"concept:{normalized}"


def edge_node_id(
    source_id: str,
    edge_type: str,
    target_id: str,
    properties: dict,
) -> str:
    return "edge:" + stable_id(source_id, edge_type, target_id, sorted(properties.items()))

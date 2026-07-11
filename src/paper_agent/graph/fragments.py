from collections.abc import Iterable

from paper_agent.domain import AnswerBundle, ChunkBundle, Paper
from paper_agent.graph.builder import GraphBuilder
from paper_agent.graph.model import GraphDocument, GraphNode, NodeType
from paper_agent.graph.workspace import GraphDelta


def build_paper_fragment(paper: Paper, chunks: ChunkBundle) -> GraphDocument:
    """Build a graph fragment for one paper and its chunks."""

    if paper.paper_id != chunks.paper_id:
        raise ValueError(
            f"paper/chunks mismatch: paper_id={paper.paper_id}, chunks.paper_id={chunks.paper_id}"
        )
    builder = GraphBuilder()
    builder.add_papers([paper])
    builder.add_chunks([chunks])
    return builder.build()


def build_answer_fragment(answer: AnswerBundle) -> GraphDocument:
    """Build a graph fragment for one saved answer bundle.

    The fragment may contain edges to existing Chunk nodes. Those chunks are expected to
    be present in the workspace effective graph when the delta is validated.
    """

    builder = GraphBuilder()
    builder.add_answers([answer])
    return builder.build()


def build_papers_fragment(papers: Iterable[Paper], chunks: Iterable[ChunkBundle]) -> GraphDocument:
    builder = GraphBuilder()
    papers_list = list(papers)
    chunks_list = list(chunks)
    builder.add_papers(papers_list)
    builder.add_chunks(chunks_list)
    return builder.build()


def delta_from_fragment(existing: GraphDocument, fragment: GraphDocument) -> GraphDelta:
    """Return the new records from `fragment` that are not already in `existing`.

    Matching IDs with different payloads are treated as conflicts instead of silently
    overwriting graph records.
    """

    existing_nodes = {node.node_id: node for node in existing.nodes}
    existing_edges = {edge.edge_id: edge for edge in existing.edges}
    added_nodes = []
    added_edges = []

    for node in fragment.nodes:
        existing_node = existing_nodes.get(node.node_id)
        if existing_node is None:
            added_nodes.append(node)
        elif not _compatible_existing_node(existing_node, node):
            raise ValueError(f"conflicting graph node already exists: {node.node_id}")

    for edge in fragment.edges:
        existing_edge = existing_edges.get(edge.edge_id)
        if existing_edge is None:
            added_edges.append(edge)
        elif existing_edge != edge:
            raise ValueError(f"conflicting graph edge already exists: {edge.edge_id}")

    return GraphDelta(added_nodes=added_nodes, added_edges=added_edges)


def _compatible_existing_node(existing: GraphNode, candidate: GraphNode) -> bool:
    if existing == candidate:
        return True
    if existing.node_type == candidate.node_type == NodeType.CONCEPT:
        existing_key = existing.properties.get("normalized")
        candidate_key = candidate.properties.get("normalized")
        return bool(existing_key and candidate_key and existing_key == candidate_key)
    return False

from collections import Counter

from pydantic import BaseModel, Field

from paper_agent.graph.model import EdgeType, GraphDocument, NodeType


class GraphValidationReport(BaseModel):
    valid: bool
    node_count: int
    edge_count: int
    errors: list[str] = Field(default_factory=list)


class GraphValidator:
    """校验证据可追溯图谱所需的可审计不变量。"""

    def validate(self, graph: GraphDocument) -> GraphValidationReport:
        errors: list[str] = []
        node_ids = [node.node_id for node in graph.nodes]
        edge_ids = [edge.edge_id for edge in graph.edges]
        errors.extend(_duplicates("duplicate node_id", node_ids))
        errors.extend(_duplicates("duplicate edge_id", edge_ids))

        node_id_set = set(node_ids)
        for edge in graph.edges:
            if edge.source_id not in node_id_set:
                errors.append(f"edge {edge.edge_id} has missing source: {edge.source_id}")
            if edge.target_id not in node_id_set:
                errors.append(f"edge {edge.edge_id} has missing target: {edge.target_id}")

        paper_ids = {node.node_id for node in graph.nodes if node.node_type == NodeType.PAPER}
        chunk_ids = {node.node_id for node in graph.nodes if node.node_type == NodeType.CHUNK}
        claim_ids = {node.node_id for node in graph.nodes if node.node_type == NodeType.CLAIM}
        concept_ids = {node.node_id for node in graph.nodes if node.node_type == NodeType.CONCEPT}
        if not paper_ids:
            errors.append("graph must contain at least one Paper node")
        if not chunk_ids:
            errors.append("graph must contain at least one Chunk node")

        paper_has_chunk_sources = {
            edge.source_id
            for edge in graph.edges
            if edge.edge_type == EdgeType.HAS_CHUNK
        }
        for paper_id in paper_ids - paper_has_chunk_sources:
            errors.append(f"paper has no HAS_CHUNK edge: {paper_id}")

        supported_claims = {
            edge.source_id
            for edge in graph.edges
            if edge.edge_type == EdgeType.SUPPORTED_BY
        }
        for claim_id in claim_ids - supported_claims:
            errors.append(f"claim has no SUPPORTED_BY edge: {claim_id}")
        for edge in graph.edges:
            if edge.edge_type == EdgeType.SUPPORTED_BY and edge.target_id not in chunk_ids:
                errors.append(f"SUPPORTED_BY must target Chunk: {edge.edge_id}")
            if edge.edge_type == EdgeType.MENTIONS and edge.target_id not in concept_ids:
                errors.append(f"MENTIONS must target Concept: {edge.edge_id}")
            if (
                edge.edge_type == EdgeType.MENTIONS
                and edge.source_id in node_id_set
                and edge.source_id not in chunk_ids | claim_ids
            ):
                errors.append(f"MENTIONS source must be Chunk or Claim: {edge.edge_id}")

        return GraphValidationReport(
            valid=not errors,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            errors=errors,
        )


def _duplicates(prefix: str, values: list[str]) -> list[str]:
    counts = Counter(values)
    return [f"{prefix}: {value}" for value, count in counts.items() if count > 1]

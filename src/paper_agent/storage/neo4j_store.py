from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from paper_agent.graph.model import GraphDocument, GraphEdge, GraphNode, NodeType


@dataclass(frozen=True)
class Neo4jWriteReport:
    node_count: int
    edge_count: int
    reset: bool


class Neo4jGraphStore:
    """将 JSONL Paper KG 物化到 Neo4j，不改变 JSONL 的事实源地位。

    节点统一使用稳定的 ``:GraphNode`` 标签，边统一使用 ``:GRAPH_EDGE``；
    领域类型存入属性，避免不安全的动态 Cypher 标签，同时保留完整图 Schema 的可查询性。
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        driver=None,
    ) -> None:
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "Install Neo4j support: pip install -e '.[storage]'"
                ) from exc
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jGraphStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
            "FOR (node:GraphNode) REQUIRE node.node_id IS UNIQUE",
            "CREATE INDEX graph_node_type IF NOT EXISTS "
            "FOR (node:GraphNode) ON (node.node_type)",
            "CREATE INDEX graph_node_paper_id IF NOT EXISTS "
            "FOR (node:GraphNode) ON (node.paper_id)",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    def replace_graph(self, graph: GraphDocument, *, reset: bool = False) -> Neo4jWriteReport:
        self.ensure_schema()
        paper_graph = paper_content_view(graph)
        nodes = [_node_payload(node) for node in paper_graph.nodes]
        edges = [_edge_payload(edge) for edge in paper_graph.edges]
        with self.driver.session(database=self.database) as session:
            if reset:
                session.run("MATCH (node:GraphNode) DETACH DELETE node").consume()
            for batch in _batches(nodes):
                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (node:GraphNode {node_id: row.node_id}) "
                    "SET node.node_type = row.node_type, node.label = row.label, "
                    "node.paper_id = row.paper_id, node.properties_json = row.properties_json",
                    rows=batch,
                ).consume()
            for batch in _batches(edges):
                session.run(
                    "UNWIND $rows AS row "
                    "MATCH (source:GraphNode {node_id: row.source_id}) "
                    "MATCH (target:GraphNode {node_id: row.target_id}) "
                    "MERGE (source)-[edge:GRAPH_EDGE {edge_id: row.edge_id}]->(target) "
                    "SET edge.edge_type = row.edge_type, edge.properties_json = row.properties_json",
                    rows=batch,
                ).consume()
        return Neo4jWriteReport(node_count=len(nodes), edge_count=len(edges), reset=reset)

    def counts(self) -> tuple[int, int]:
        with self.driver.session(database=self.database) as session:
            record = session.run(
                "MATCH (node:GraphNode) "
                "OPTIONAL MATCH ()-[edge:GRAPH_EDGE]->() "
                "RETURN count(DISTINCT node) AS nodes, count(DISTINCT edge) AS edges"
            ).single()
        return int(record["nodes"]), int(record["edges"])

    def papers(self) -> list[dict[str, str]]:
        with self.driver.session(database=self.database) as session:
            records = session.run(
                "MATCH (paper:GraphNode {node_type: 'Paper'}) "
                "RETURN paper.paper_id AS paper_id, paper.label AS title ORDER BY paper_id"
            )
            return [dict(record) for record in records]

    def concept_papers(self, concept: str, limit: int = 20) -> list[dict[str, str]]:
        with self.driver.session(database=self.database) as session:
            records = session.run(
                "MATCH (concept:GraphNode {node_type: 'Concept'}) "
                "WHERE toLower(concept.label) = toLower($concept) "
                "MATCH (chunk:GraphNode {node_type: 'Chunk'})"
                "-[mention:GRAPH_EDGE {edge_type: 'MENTIONS'}]->(concept) "
                "MATCH (paper:GraphNode {node_id: 'paper:' + chunk.paper_id}) "
                "RETURN DISTINCT paper.paper_id AS paper_id, paper.label AS title "
                "ORDER BY paper_id LIMIT $limit",
                concept=concept,
                limit=limit,
            )
            return [dict(record) for record in records]


def _node_payload(node: GraphNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type.value,
        "label": node.label,
        "paper_id": str(node.properties.get("paper_id", "")) or None,
        "properties_json": json.dumps(node.properties, ensure_ascii=False, sort_keys=True),
    }


def _edge_payload(edge: GraphEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "edge_type": edge.edge_type.value,
        "properties_json": json.dumps(edge.properties, ensure_ascii=False, sort_keys=True),
    }


def _batches(rows: list[dict[str, Any]], size: int = 500) -> list[list[dict[str, Any]]]:
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def paper_content_view(graph: GraphDocument) -> GraphDocument:
    """物化 Paper KG 前删除历史回答与问题工件。"""

    allowed_types = {NodeType.PAPER, NodeType.SECTION, NodeType.CHUNK, NodeType.CONCEPT}
    nodes = [node for node in graph.nodes if node.node_type in allowed_types]
    node_ids = {node.node_id for node in nodes}
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in node_ids and edge.target_id in node_ids
    ]
    return GraphDocument(nodes=nodes, edges=edges)

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("Neo4j store tests require pydantic v2", allow_module_level=True)

from paper_agent.graph import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType
from paper_agent.storage.neo4j_store import Neo4jGraphStore, paper_content_view


class FakeResult:
    def __init__(self, rows=None, single=None):
        self.rows = rows or []
        self._single = single

    def consume(self):
        return self

    def single(self):
        return self._single

    def __iter__(self):
        return iter(self.rows)


class FakeSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, statement, **kwargs):
        self.calls.append((statement, kwargs))
        if "count(DISTINCT node)" in statement:
            return FakeResult(single={"nodes": 2, "edges": 1})
        return FakeResult()


class FakeDriver:
    def __init__(self):
        self.calls = []
        self.closed = False

    def session(self, **_kwargs):
        return FakeSession(self.calls)

    def close(self):
        self.closed = True


def sample_graph():
    return GraphDocument(
        nodes=[
            GraphNode(
                node_id="paper:paper_a",
                node_type=NodeType.PAPER,
                label="Paper A",
                properties={"paper_id": "paper_a", "title": "Paper A"},
            ),
            GraphNode(
                node_id="chunk:chunk_a",
                node_type=NodeType.CHUNK,
                label="chunk_a",
                properties={"paper_id": "paper_a"},
            ),
        ],
        edges=[
            GraphEdge(
                edge_id="edge:contains",
                source_id="paper:paper_a",
                target_id="chunk:chunk_a",
                edge_type=EdgeType.HAS_CHUNK,
            )
        ],
    )


def test_neo4j_store_materializes_stable_node_and_edge_ids_without_network():
    driver = FakeDriver()
    store = Neo4jGraphStore("bolt://unused", "neo4j", "secret", driver=driver)

    report = store.replace_graph(sample_graph(), reset=True)
    nodes, edges = store.counts()
    store.close()

    statements = [statement for statement, _ in driver.calls]
    node_write = next(kwargs for statement, kwargs in driver.calls if "MERGE (node:GraphNode" in statement)
    edge_write = next(kwargs for statement, kwargs in driver.calls if "MERGE (source)-[edge:GRAPH_EDGE" in statement)
    assert any("CREATE CONSTRAINT graph_node_id" in statement for statement in statements)
    assert any("DETACH DELETE" in statement for statement in statements)
    assert node_write["rows"][0]["node_id"] == "paper:paper_a"
    assert edge_write["rows"][0]["edge_id"] == "edge:contains"
    assert report.node_count == 2
    assert report.edge_count == 1
    assert report.reset
    assert (nodes, edges) == (2, 1)
    assert driver.closed


def test_neo4j_materialization_excludes_legacy_answer_artifacts():
    graph = sample_graph().model_copy(
        update={
            "nodes": [
                *sample_graph().nodes,
                GraphNode(node_id="answer:old", node_type=NodeType.ANSWER, label="old answer"),
            ]
        }
    )

    view = paper_content_view(graph)

    assert {node.node_type for node in view.nodes} == {NodeType.PAPER, NodeType.CHUNK}
    assert all(node.node_id != "answer:old" for node in view.nodes)

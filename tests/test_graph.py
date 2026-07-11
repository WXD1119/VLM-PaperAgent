import pytest
import pydantic

if not hasattr(pydantic, "model_validator"):
    pytest.skip("graph tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkBundle,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    Paper,
    RetrievalChunk,
)
from paper_agent.graph.concepts import extract_concepts
from paper_agent.graph import (
    EdgeType,
    GraphBuilder,
    GraphDocument,
    GraphEdge,
    GraphNode,
    GraphQuery,
    GraphValidator,
    GraphDelta,
    LocalGraphWorkspaceStore,
    NodeType,
    diff_graphs,
    write_graph_jsonl,
)


def sample_paper() -> Paper:
    return Paper(
        paper_id="paper-1",
        title="Test Paper",
        source_path="data/raw/test.pdf",
        sha256="abc",
        elements=[],
    )


def test_concept_extractor_trims_boundaries_and_filters_author_names():
    labels = [
        mention.label
        for mention in extract_concepts(
            "Flow Matching The method follows De Bortoli and Ben-Hamu for "
            "Diffusion-based full-atom design with CFM and Q-Former cross-attention."
        )
    ]

    assert "Flow Matching The" not in labels
    assert "De Bortoli" not in labels
    assert "Ben-Hamu" not in labels
    assert "flow matching" in labels
    assert "CFM" in labels


def sample_chunks() -> ChunkBundle:
    return ChunkBundle(
        paper_id="paper-1",
        parents=[],
        children=[
            RetrievalChunk(
                chunk_id="chunk-1",
                parent_chunk_id="parent-1",
                paper_id="paper-1",
                kind=ChunkKind.TEXT,
                content="Q-Former uses learnable queries.",
                element_ids=["el-1"],
                pages=[1],
                section_path=["Title", "Method"],
            )
        ],
    )


def sample_answer() -> AnswerBundle:
    return AnswerBundle(
        evidence_pack=EvidencePack(
            query="How does Q-Former work?",
            items=[
                EvidenceItem(
                    evidence_id="E1",
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    kind=ChunkKind.TEXT,
                    pages=[1],
                    section_path=["Title", "Method"],
                    content="Q-Former uses learnable queries.",
                )
            ],
        ),
        answer=GroundedAnswer(
            answer="Q-Former uses learnable queries.",
            claims=[
                AnswerClaim(
                    text="Q-Former uses learnable queries.",
                    evidence_ids=["E1"],
                )
            ],
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=1,
            cited_claim_count=1,
        ),
        generator_model="fake-model",
    )


def test_graph_builder_links_claims_to_existing_chunks():
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    graph = builder.build()

    report = GraphValidator().validate(graph)

    assert report.valid
    assert any(node.node_type.value == "Paper" for node in graph.nodes)
    assert any(node.node_type.value == "Section" for node in graph.nodes)
    assert any(node.node_type.value == "Chunk" for node in graph.nodes)
    assert any(node.node_type.value == "Claim" for node in graph.nodes)
    assert any(
        node.node_type == NodeType.CONCEPT and node.label == "Q-Former"
        for node in graph.nodes
    )
    assert any(edge.edge_type == EdgeType.SUPPORTED_BY for edge in graph.edges)
    assert any(edge.edge_type == EdgeType.MENTIONS for edge in graph.edges)


def test_graph_validator_rejects_missing_edge_target():
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    graph = builder.build()
    graph = GraphDocument(
        nodes=graph.nodes,
        edges=graph.edges
        + [
            GraphEdge(
                edge_id="edge:bad",
                source_id=graph.nodes[0].node_id,
                target_id="chunk:missing",
                edge_type=EdgeType.CITES,
            )
        ],
    )

    report = GraphValidator().validate(graph)

    assert not report.valid
    assert any("missing target" in error for error in report.errors)


def test_graph_validator_rejects_mentions_to_non_concept():
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    graph = builder.build()
    chunk_nodes = [node for node in graph.nodes if node.node_type == NodeType.CHUNK]
    graph = GraphDocument(
        nodes=graph.nodes,
        edges=graph.edges
        + [
            GraphEdge(
                edge_id="edge:bad-mention",
                source_id=chunk_nodes[0].node_id,
                target_id=chunk_nodes[0].node_id,
                edge_type=EdgeType.MENTIONS,
            )
        ],
    )

    report = GraphValidator().validate(graph)

    assert not report.valid
    assert any("MENTIONS must target Concept" in error for error in report.errors)


def test_graph_query_lists_papers_chunks_claim_supports_and_search():
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    query = GraphQuery(builder.build())

    papers = query.papers()
    chunks = query.chunks_for_paper("paper-1")
    supports = query.claim_supports()
    matches = query.search_nodes("learnable", node_types=[NodeType.CHUNK, NodeType.CLAIM])
    neighborhood = query.concept_neighborhood("Q-Former")

    assert [paper.properties["paper_id"] for paper in papers] == ["paper-1"]
    assert [chunk.properties["chunk_id"] for chunk in chunks] == ["chunk-1"]
    assert supports[0][0].node_type == NodeType.CLAIM
    assert supports[0][1][0].properties["chunk_id"] == "chunk-1"
    assert {node.node_type for node in matches} == {NodeType.CHUNK, NodeType.CLAIM}
    assert neighborhood.concept is not None
    assert neighborhood.concept.label == "Q-Former"
    assert [paper.properties["paper_id"] for paper in neighborhood.papers] == ["paper-1"]
    assert neighborhood.claims[0][0].properties["text"] == "Q-Former uses learnable queries."
    assert neighborhood.chunks[0].properties["chunk_id"] == "chunk-1"


def test_graph_query_returns_candidates_for_ambiguous_concept_keyword():
    graph = GraphDocument(
        nodes=[
            GraphNode(
                node_id="concept:vision-language-alignment",
                node_type=NodeType.CONCEPT,
                label="vision-language alignment",
                properties={"aliases": ["alignment"]},
            ),
            GraphNode(
                node_id="concept:sequence-alignment",
                node_type=NodeType.CONCEPT,
                label="sequence alignment",
                properties={"aliases": ["alignment"]},
            ),
        ],
        edges=[],
    )
    query = GraphQuery(graph)

    neighborhood = query.concept_neighborhood("alignment")

    assert neighborhood.concept is None
    assert [candidate.label for candidate in neighborhood.candidates] == [
        "sequence alignment",
        "vision-language alignment",
    ]


def test_graph_workspace_fork_loads_base_graph_without_rewriting_it(tmp_path):
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    base_graph = builder.build()
    base_dir = tmp_path / "base_graph"
    write_graph_jsonl(base_graph, base_dir)
    base_nodes_before = (base_dir / "nodes.jsonl").read_text(encoding="utf-8")

    workspace_dir = tmp_path / "workspace"
    store = LocalGraphWorkspaceStore(workspace_dir)
    workspace = store.create_fork(
        base_graph_path=base_dir,
        workspace_id="ws-test",
        owner_id="wxd",
        name="WXD Test Workspace",
    )
    effective_graph = store.load_effective_graph()
    report = GraphValidator().validate(effective_graph)

    assert workspace.workspace_id == "ws-test"
    assert workspace.head_commit_id is not None
    assert (workspace_dir / "workspace.json").exists()
    assert len(store.commit_chain()) == 1
    assert report.valid
    assert len(effective_graph.nodes) == len(base_graph.nodes)
    assert len(effective_graph.edges) == len(base_graph.edges)
    assert (base_dir / "nodes.jsonl").read_text(encoding="utf-8") == base_nodes_before


def test_graph_workspace_delta_tombstone_hides_node_and_incident_edges(tmp_path):
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    base_graph = builder.build()
    base_dir = tmp_path / "base_graph"
    write_graph_jsonl(base_graph, base_dir)
    chunk = next(node for node in base_graph.nodes if node.node_type == NodeType.CHUNK)

    store = LocalGraphWorkspaceStore(tmp_path / "workspace")
    store.create_fork(
        base_graph_path=base_dir,
        workspace_id="ws-test",
        owner_id="wxd",
    )
    store.commit_delta(
        GraphDelta(removed_node_ids=[chunk.node_id]),
        author_id="wxd",
        message="hide chunk",
    )
    effective_graph = store.load_effective_graph()
    effective_node_ids = {node.node_id for node in effective_graph.nodes}

    assert chunk.node_id not in effective_node_ids
    assert all(
        edge.source_id != chunk.node_id and edge.target_id != chunk.node_id
        for edge in effective_graph.edges
    )


def test_graph_workspace_commit_delta_adds_records_and_diff_keeps_base_unchanged(tmp_path):
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    base_graph = builder.build()
    base_dir = tmp_path / "base_graph"
    write_graph_jsonl(base_graph, base_dir)
    base_edges_before = (base_dir / "edges.jsonl").read_text(encoding="utf-8")
    chunk = next(node for node in base_graph.nodes if node.node_type == NodeType.CHUNK)
    concept = GraphNode(
        node_id="concept:private-note",
        node_type=NodeType.CONCEPT,
        label="private note",
        properties={"normalized": "private-note", "aliases": ["private note"]},
    )
    mention = GraphEdge(
        edge_id="edge:private-note-mentioned",
        source_id=chunk.node_id,
        target_id=concept.node_id,
        edge_type=EdgeType.MENTIONS,
        properties={"source_kind": "chunk", "mention": "private note"},
    )

    store = LocalGraphWorkspaceStore(tmp_path / "workspace")
    store.create_fork(
        base_graph_path=base_dir,
        workspace_id="ws-test",
        owner_id="wxd",
    )
    commit = store.commit_delta(
        GraphDelta(added_nodes=[concept], added_edges=[mention]),
        author_id="wxd",
        message="add private concept",
    )
    effective_graph = store.load_effective_graph()
    diff = diff_graphs(base_graph, effective_graph)
    report = GraphValidator().validate(effective_graph)

    assert commit.stats == {
        "added_nodes": 1,
        "added_edges": 1,
        "removed_nodes": 0,
        "removed_edges": 0,
    }
    assert report.valid
    assert "concept:private-note" in diff.added_node_ids
    assert "edge:private-note-mentioned" in diff.added_edge_ids
    assert diff.added_node_types == {"Concept": 1}
    assert diff.added_edge_types == {"MENTIONS": 1}
    assert (base_dir / "edges.jsonl").read_text(encoding="utf-8") == base_edges_before


def test_graph_workspace_preview_delta_detects_invalid_effective_graph(tmp_path):
    builder = GraphBuilder()
    builder.add_papers([sample_paper()])
    builder.add_chunks([sample_chunks()])
    builder.add_answers([sample_answer()])
    base_graph = builder.build()
    base_dir = tmp_path / "base_graph"
    write_graph_jsonl(base_graph, base_dir)
    chunk = next(node for node in base_graph.nodes if node.node_type == NodeType.CHUNK)
    bad_edge = GraphEdge(
        edge_id="edge:bad-private",
        source_id=chunk.node_id,
        target_id=chunk.node_id,
        edge_type=EdgeType.MENTIONS,
    )

    store = LocalGraphWorkspaceStore(tmp_path / "workspace")
    store.create_fork(
        base_graph_path=base_dir,
        workspace_id="ws-test",
        owner_id="wxd",
    )
    preview = store.preview_delta(GraphDelta(added_edges=[bad_edge]))
    report = GraphValidator().validate(preview)

    assert not report.valid
    assert any("MENTIONS must target Concept" in error for error in report.errors)

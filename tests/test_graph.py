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
    NodeType,
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
    assert any(node.node_type == NodeType.CONCEPT and node.label == "Q-Former" for node in graph.nodes)
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

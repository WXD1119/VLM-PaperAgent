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
from paper_agent.graph import EdgeType, GraphBuilder, GraphDocument, GraphEdge, GraphValidator


def sample_paper() -> Paper:
    return Paper(
        paper_id="paper-1",
        title="Test Paper",
        source_path="data/raw/test.pdf",
        sha256="abc",
        elements=[],
    )


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
    assert any(edge.edge_type == EdgeType.SUPPORTED_BY for edge in graph.edges)


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

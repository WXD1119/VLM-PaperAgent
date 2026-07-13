import pytest

from paper_agent.domain.chunk import ChunkKind, RetrievalChunk
from paper_agent.evaluation import RetrievalCase, evaluate_bm25, evaluate_retriever
from paper_agent.evaluation.retrieval import hit_at_k, ndcg_at_k
from paper_agent.retrieval import BM25Index


def chunk(chunk_id: str, content: str) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        parent_chunk_id="parent",
        paper_id="paper",
        kind=ChunkKind.TEXT,
        content=content,
        element_ids=[chunk_id],
        pages=[1],
        section_path=["Method"],
    )


def test_evaluate_bm25_computes_macro_metrics() -> None:
    index = BM25Index([chunk("a", "rare qformer"), chunk("b", "generic model")])
    cases = [
        RetrievalCase(
            query_id="q1",
            query="qformer",
            relevant_chunk_ids={"a"},
        )
    ]
    result = evaluate_bm25(index, cases)
    assert result.mean_hit_at_1 == 1.0
    assert result.mean_hit_at_5 == 1.0
    assert result.macro_recall_at_1 == 1.0
    assert result.macro_recall_at_5 == 1.0
    assert result.mean_reciprocal_rank == 1.0
    assert result.mean_ndcg_at_5 == 1.0


def test_evaluate_bm25_requires_cases() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_bm25(BM25Index([]), [])


def test_generic_retriever_uses_the_same_metric_contract() -> None:
    index = BM25Index([chunk("a", "semantic evidence")])
    case = RetrievalCase(
        query_id="q",
        query="semantic",
        relevant_chunk_ids={"a"},
    )
    assert evaluate_retriever(index, [case]).mean_reciprocal_rank == 1.0


def test_ndcg_discounts_late_hits_and_normalizes_multiple_relevant_items() -> None:
    score = ndcg_at_k(["noise", "a", "b"], {"a", "b"}, 3)
    assert score == pytest.approx((1 / 1.5849625 + 1 / 2) / (1 + 1 / 1.5849625))


def test_hit_at_k_is_binary_for_any_relevant_match() -> None:
    assert hit_at_k(["noise", "a"], {"a", "b"}, 1) == 0.0
    assert hit_at_k(["noise", "a"], {"a", "b"}, 2) == 1.0

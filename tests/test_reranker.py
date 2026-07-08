from types import SimpleNamespace

import pytest

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval import RerankedRetriever


def hit(chunk_id: str, content: str, score: float):
    return SimpleNamespace(
        chunk_id=chunk_id,
        paper_id="paper",
        kind=ChunkKind.TEXT,
        score=score,
        sparse_rank=1,
        dense_rank=2,
        pages=[1],
        section_path=["Method"],
        content=content,
        context="context",
    )


class StaticRetriever:
    def __init__(self, hits) -> None:
        self.hits = hits

    def search(self, query, top_k=5, paper_id=None, kind=None):
        return self.hits[:top_k]


class KeywordReranker:
    model_name = "fake"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 if "target" in document else 0.0 for document in documents]


def test_reranker_promotes_relevant_candidate_and_preserves_provenance() -> None:
    retriever = StaticRetriever([hit("first", "noise", 0.9), hit("second", "target", 0.5)])
    result = RerankedRetriever(retriever, KeywordReranker(), candidate_k=2).search(
        "query", top_k=1
    )
    assert result[0].chunk_id == "second"
    assert result[0].retrieval_rank == 2
    assert result[0].retrieval_score == 0.5
    assert result[0].sparse_rank == 1


def test_reranker_rejects_score_count_mismatch() -> None:
    class BrokenReranker(KeywordReranker):
        def score(self, query: str, documents: list[str]) -> list[float]:
            return []

    reranked = RerankedRetriever(StaticRetriever([hit("a", "target", 1.0)]), BrokenReranker())
    with pytest.raises(ValueError, match="one score per candidate"):
        reranked.search("query")

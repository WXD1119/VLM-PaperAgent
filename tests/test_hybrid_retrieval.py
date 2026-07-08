from types import SimpleNamespace

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval.hybrid import HybridRetriever


def hit(chunk_id: str):
    return SimpleNamespace(
        chunk_id=chunk_id,
        paper_id="paper",
        kind=ChunkKind.TEXT,
        pages=[1],
        section_path=["Method"],
        content=chunk_id,
        context="",
    )


class StaticIndex:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def search(self, query, top_k=5, paper_id=None, kind=None):
        return [hit(chunk_id) for chunk_id in self.ids[:top_k]]


def test_rrf_rewards_chunks_present_in_both_rankings() -> None:
    hybrid = HybridRetriever(
        StaticIndex(["sparse-only", "shared"]),
        StaticIndex(["dense-only", "shared"]),
        rrf_k=60,
        candidate_k=10,
    )
    results = hybrid.search("query", top_k=3)
    assert results[0].chunk_id == "shared"
    assert results[0].sparse_rank == 2
    assert results[0].dense_rank == 2


def test_rrf_keeps_single_source_candidates_and_provenance() -> None:
    hybrid = HybridRetriever(StaticIndex(["a"]), StaticIndex(["b"]), candidate_k=2)
    results = hybrid.search("query", top_k=2)
    assert {result.chunk_id for result in results} == {"a", "b"}
    assert any(result.sparse_rank == 1 and result.dense_rank is None for result in results)
    assert any(result.dense_rank == 1 and result.sparse_rank is None for result in results)

from typing import Protocol

from pydantic import BaseModel, Field

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval.fusion import reciprocal_rank_fusion


class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list: ...


class HybridHit(BaseModel):
    chunk_id: str
    paper_id: str
    kind: ChunkKind
    score: float = Field(gt=0)
    sparse_rank: int | None = None
    dense_rank: int | None = None
    pages: list[int]
    section_path: list[str]
    content: str
    context: str = ""


class HybridRetriever:
    """Fuse sparse and dense rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        sparse: SearchIndex,
        dense: SearchIndex,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[HybridHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidate_k = max(top_k, self.candidate_k)
        sparse_hits = self.sparse.search(query, candidate_k, paper_id, kind)
        dense_hits = self.dense.search(query, candidate_k, paper_id, kind)
        sparse_ids = [hit.chunk_id for hit in sparse_hits]
        dense_ids = [hit.chunk_id for hit in dense_hits]
        fused = reciprocal_rank_fusion([sparse_ids, dense_ids], k=self.rrf_k)

        hit_by_id = {hit.chunk_id: hit for hit in dense_hits}
        hit_by_id.update({hit.chunk_id: hit for hit in sparse_hits})
        sparse_rank = {chunk_id: rank for rank, chunk_id in enumerate(sparse_ids, start=1)}
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
        results: list[HybridHit] = []
        for chunk_id, score in fused[:top_k]:
            source = hit_by_id[chunk_id]
            results.append(
                HybridHit(
                    chunk_id=chunk_id,
                    paper_id=source.paper_id,
                    kind=source.kind,
                    score=score,
                    sparse_rank=sparse_rank.get(chunk_id),
                    dense_rank=dense_rank.get(chunk_id),
                    pages=source.pages,
                    section_path=source.section_path,
                    content=source.content,
                    context=source.context,
                )
            )
        return results

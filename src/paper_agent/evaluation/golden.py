from statistics import mean
from typing import Protocol

from pydantic import BaseModel, Field

from paper_agent.domain.chunk import ChunkKind
from paper_agent.evaluation.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank


class RetrievalIndex(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list: ...


class RetrievalCase(BaseModel):
    query_id: str
    query: str
    relevant_chunk_ids: set[str] = Field(min_length=1)
    paper_id: str | None = None
    kind: ChunkKind | None = None
    category: str = "unspecified"
    annotator: str = ""
    expected_answer: str = ""
    notes: str = ""


class RetrievalCaseResult(BaseModel):
    query_id: str
    retrieved_chunk_ids: list[str]
    recall_at_1: float
    recall_at_5: float
    reciprocal_rank: float
    ndcg_at_5: float


class RetrievalEvaluation(BaseModel):
    cases: list[RetrievalCaseResult]
    macro_recall_at_1: float
    macro_recall_at_5: float
    mean_reciprocal_rank: float
    mean_ndcg_at_5: float


def evaluate_bm25(
    index: RetrievalIndex,
    cases: list[RetrievalCase],
    top_k: int = 5,
) -> RetrievalEvaluation:
    return evaluate_retriever(index, cases, top_k)


def evaluate_retriever(
    index: RetrievalIndex,
    cases: list[RetrievalCase],
    top_k: int = 5,
) -> RetrievalEvaluation:
    if not cases:
        raise ValueError("at least one retrieval case is required")
    results: list[RetrievalCaseResult] = []
    for case in cases:
        hits = index.search(
            case.query,
            top_k=top_k,
            paper_id=case.paper_id,
            kind=case.kind,
        )
        retrieved = [hit.chunk_id for hit in hits]
        results.append(
            RetrievalCaseResult(
                query_id=case.query_id,
                retrieved_chunk_ids=retrieved,
                recall_at_1=recall_at_k(retrieved, case.relevant_chunk_ids, 1),
                recall_at_5=recall_at_k(retrieved, case.relevant_chunk_ids, 5),
                reciprocal_rank=reciprocal_rank(retrieved, case.relevant_chunk_ids),
                ndcg_at_5=ndcg_at_k(retrieved, case.relevant_chunk_ids, 5),
            )
        )
    return RetrievalEvaluation(
        cases=results,
        macro_recall_at_1=mean(result.recall_at_1 for result in results),
        macro_recall_at_5=mean(result.recall_at_5 for result in results),
        mean_reciprocal_rank=mean(result.reciprocal_rank for result in results),
        mean_ndcg_at_5=mean(result.ndcg_at_5 for result in results),
    )

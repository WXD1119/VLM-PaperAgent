from typing import Protocol

from pydantic import BaseModel

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval.hybrid import SearchIndex


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Sentence Transformers cross-encoder adapter with offline model support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 2048,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies: pip install -e '.[retrieval]'"
            ) from exc
        self._model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            local_files_only=local_files_only,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self.model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=len(documents) > self.batch_size,
        )
        values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return [float(value[0] if isinstance(value, list) else value) for value in values]


class RerankedHit(BaseModel):
    chunk_id: str
    paper_id: str
    kind: ChunkKind
    score: float
    retrieval_score: float
    retrieval_rank: int
    sparse_rank: int | None = None
    dense_rank: int | None = None
    pages: list[int]
    section_path: list[str]
    content: str
    context: str = ""


class RerankedRetriever:
    """Retrieve candidates first, then reorder them with a cross encoder."""

    def __init__(
        self,
        retriever: SearchIndex,
        reranker: Reranker,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[RerankedHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = self.retriever.search(
            query,
            top_k=max(top_k, self.candidate_k),
            paper_id=paper_id,
            kind=kind,
        )
        documents = [self._document(candidate) for candidate in candidates]
        scores = self.reranker.score(query, documents)
        if len(scores) != len(candidates):
            raise ValueError("reranker must return exactly one score per candidate")
        ranked = sorted(
            enumerate(zip(candidates, scores), start=1),
            key=lambda item: (-item[1][1], item[0]),
        )
        results: list[RerankedHit] = []
        for retrieval_rank, (candidate, reranker_score) in ranked[:top_k]:
            results.append(
                RerankedHit(
                    chunk_id=candidate.chunk_id,
                    paper_id=candidate.paper_id,
                    kind=candidate.kind,
                    score=reranker_score,
                    retrieval_score=candidate.score,
                    retrieval_rank=retrieval_rank,
                    sparse_rank=getattr(candidate, "sparse_rank", None),
                    dense_rank=getattr(candidate, "dense_rank", None),
                    pages=candidate.pages,
                    section_path=candidate.section_path,
                    content=candidate.content,
                    context=candidate.context,
                )
            )
        return results

    @staticmethod
    def _document(candidate) -> str:
        section = " > ".join(candidate.section_path)
        return "\n".join(part for part in (section, candidate.context, candidate.content) if part)

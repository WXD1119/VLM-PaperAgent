import math
from dataclasses import dataclass

from pydantic import BaseModel, Field

from paper_agent.domain.chunk import ChunkKind, RetrievalChunk
from paper_agent.retrieval.embedding import EmbeddingEncoder


class DenseHit(BaseModel):
    chunk_id: str
    paper_id: str
    kind: ChunkKind
    score: float = Field(ge=-1, le=1)
    pages: list[int]
    section_path: list[str]
    content: str
    context: str = ""


@dataclass
class _DenseRecord:
    chunk: RetrievalChunk
    vector: list[float]


class InMemoryDenseIndex:
    """Reference cosine index used for tests and small local experiments."""

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self.encoder = encoder
        self._records: list[_DenseRecord] = []

    def upsert(self, chunks: list[RetrievalChunk]) -> None:
        vectors = self.encoder.encode_documents([chunk.embedding_text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("encoder returned a different number of vectors than documents")
        existing = {record.chunk.chunk_id: record for record in self._records}
        for chunk, vector in zip(chunks, vectors, strict=True):
            _validate_dimension(vector, self.encoder.dimension)
            existing[chunk.chunk_id] = _DenseRecord(chunk=chunk, vector=vector)
        self._records = list(existing.values())

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[DenseHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vectors = self.encoder.encode_queries([query])
        if len(vectors) != 1:
            raise ValueError("encoder must return exactly one query vector")
        query_vector = vectors[0]
        _validate_dimension(query_vector, self.encoder.dimension)
        scored: list[tuple[float, RetrievalChunk]] = []
        for record in self._records:
            if paper_id is not None and record.chunk.paper_id != paper_id:
                continue
            if kind is not None and record.chunk.kind != kind:
                continue
            scored.append((_cosine(query_vector, record.vector), record.chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [_hit(chunk, score) for score, chunk in scored[:top_k]]


def _hit(chunk: RetrievalChunk, score: float) -> DenseHit:
    return DenseHit(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        kind=chunk.kind,
        score=max(-1.0, min(1.0, score)),
        pages=chunk.pages,
        section_path=chunk.section_path,
        content=chunk.content,
        context=chunk.context,
    )


def _validate_dimension(vector: list[float], expected: int) -> None:
    if len(vector) != expected:
        raise ValueError(f"embedding dimension mismatch: expected {expected}, got {len(vector)}")


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

from paper_agent.domain.chunk import ChunkKind, RetrievalChunk
from paper_agent.retrieval.dense import InMemoryDenseIndex


class FakeEncoder:
    model_name = "fake-v1"
    dimension = 2

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [1.0, 0.0] if "vision" in lowered else [0.0, 1.0]


def chunk(chunk_id: str, content: str, paper_id: str = "paper_x") -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        parent_chunk_id="parent",
        paper_id=paper_id,
        kind=ChunkKind.TEXT,
        content=content,
        element_ids=[chunk_id],
        pages=[1],
        section_path=["Method"],
    )


def test_dense_index_ranks_by_cosine_and_preserves_evidence() -> None:
    index = InMemoryDenseIndex(FakeEncoder())
    index.upsert([chunk("vision", "vision encoder"), chunk("flow", "flow matching")])
    hits = index.search("visual vision representation")
    assert hits[0].chunk_id == "vision"
    assert hits[0].pages == [1]
    assert hits[0].section_path == ["Method"]


def test_dense_index_upsert_is_idempotent_and_filters_paper() -> None:
    index = InMemoryDenseIndex(FakeEncoder())
    index.upsert([chunk("same", "vision", "paper_a")])
    index.upsert([chunk("same", "flow", "paper_a"), chunk("other", "vision", "paper_b")])
    hits = index.search("flow", paper_id="paper_a")
    assert [hit.chunk_id for hit in hits] == ["same"]


def test_dense_index_rejects_invalid_vector_dimension() -> None:
    encoder = FakeEncoder()
    encoder.dimension = 3
    index = InMemoryDenseIndex(encoder)
    try:
        index.upsert([chunk("x", "vision")])
        raise AssertionError("dimension mismatch was accepted")
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)

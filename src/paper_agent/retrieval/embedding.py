from typing import Protocol


class EmbeddingEncoder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Lazy Sentence Transformers encoder with normalized dense vectors."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 8192,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies: pip install -e '.[retrieval]'"
            ) from exc
        self._model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )
        self.model.max_seq_length = max_length
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self._dimension = int(dimension_getter())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        # BGE-M3 does not require a query instruction prefix.
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
        )
        return vectors.astype("float32").tolist()

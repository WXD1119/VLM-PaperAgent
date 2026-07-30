import json
from pathlib import Path

from paper_agent.domain.chunk import ChunkKind, RetrievalChunk
from paper_agent.retrieval.dense import DenseHit
from paper_agent.retrieval.embedding import EmbeddingEncoder


class ChromaVectorStore:
    """使用调用方计算向量的持久化 Chroma 适配器。"""

    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        encoder: EmbeddingEncoder,
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies: pip install -e '.[retrieval]'"
            ) from exc
        self.encoder = encoder
        self.client = chromadb.PersistentClient(path=str(path))
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
            except Exception as exc:
                if "does not exist" not in str(exc).lower():
                    raise
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "embedding_model": encoder.model_name,
                "embedding_dimension": encoder.dimension,
            },
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        metadata = self.collection.metadata or {}
        stored_model = metadata.get("embedding_model")
        stored_dimension = metadata.get("embedding_dimension")
        if stored_model != encoder.model_name or stored_dimension != encoder.dimension:
            raise ValueError(
                "Chroma collection embedding configuration mismatch: "
                f"stored=({stored_model}, {stored_dimension}), "
                f"requested=({encoder.model_name}, {encoder.dimension})"
            )

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(self, chunks: list[RetrievalChunk], batch_size: int = 32) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.encoder.encode_documents(
                [chunk.embedding_text for chunk in batch]
            )
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=[chunk.content for chunk in batch],
                metadatas=[_metadata(chunk, self.encoder.model_name) for chunk in batch],
            )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[DenseHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = self.encoder.encode_queries([query])
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            where=_where(paper_id, kind),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[DenseHit] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            hits.append(
                DenseHit(
                    chunk_id=chunk_id,
                    paper_id=metadata["paper_id"],
                    kind=ChunkKind(metadata["kind"]),
                    score=max(-1.0, min(1.0, 1.0 - float(distance))),
                    pages=json.loads(metadata["pages_json"]),
                    section_path=json.loads(metadata["section_path_json"]),
                    content=document or "",
                    context=metadata.get("context", ""),
                )
            )
        return hits


def _metadata(chunk: RetrievalChunk, model_name: str) -> dict:
    return {
        "paper_id": chunk.paper_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "kind": chunk.kind.value,
        "pages_json": json.dumps(chunk.pages),
        "section_path_json": json.dumps(chunk.section_path, ensure_ascii=False),
        "context": chunk.context,
        "embedding_model": model_name,
    }


def _where(paper_id: str | None, kind: ChunkKind | None) -> dict | None:
    filters: list[dict] = []
    if paper_id is not None:
        filters.append({"paper_id": paper_id})
    if kind is not None:
        filters.append({"kind": kind.value})
    if not filters:
        return None
    return filters[0] if len(filters) == 1 else {"$and": filters}

"""Scoped hybrid retrieval for unpromoted-paper workspaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from typing import Protocol

from paper_agent.domain import ChunkKind, RetrievalChunk
from paper_agent.retrieval import BM25Index, HybridRetriever, chunks_from_bundles, load_chunk_bundles
from paper_agent.retrieval.embedding import EmbeddingEncoder
from paper_agent.storage import ChromaVectorStore
from paper_agent.temporary_workspace.models import TemporaryPaperWorkspace


class WorkspaceStore(Protocol):
    def load(self, *, user_id: str, session_id: str, workspace_id: str) -> TemporaryPaperWorkspace: ...


DenseFactory = Callable[[Path, str, EmbeddingEncoder], object]


class TemporaryWorkspaceIndexManager:
    """Build and query an index whose corpus is exactly one temporary workspace.

    The index directory and Chroma collection are derived from a validated
    workspace ID.  Callers must provide user/session identity on every query,
    so an ID alone never grants access to uploaded-paper content.
    """

    def __init__(
        self,
        store: WorkspaceStore,
        encoder: EmbeddingEncoder,
        *,
        root: str | Path = "artifacts/temporary_indexes",
        dense_factory: DenseFactory | None = None,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.root = Path(root)
        self.dense_factory = dense_factory or (lambda path, name, current_encoder: ChromaVectorStore(path, name, current_encoder))
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self._retrievers: dict[str, HybridRetriever] = {}

    def build(self, *, user_id: str, session_id: str, workspace_id: str) -> int:
        workspace = self.store.load(user_id=user_id, session_id=session_id, workspace_id=workspace_id)
        chunks = self._load_workspace_chunks(workspace)
        allowed = {item.paper_id for item in workspace.papers}
        if any(chunk.paper_id not in allowed for chunk in chunks):
            raise ValueError("temporary chunks contain a paper outside the workspace")
        path = self.root / workspace.workspace_id
        dense = self.dense_factory(path, self._collection_name(workspace), self.encoder)
        dense.upsert(chunks)
        self._retrievers[workspace.workspace_id] = HybridRetriever(
            BM25Index(chunks), dense, rrf_k=self.rrf_k, candidate_k=self.candidate_k
        )
        return len(chunks)

    def search(
        self,
        *,
        user_id: str,
        session_id: str,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[object]:
        workspace = self.store.load(user_id=user_id, session_id=session_id, workspace_id=workspace_id)
        allowed = {item.paper_id for item in workspace.papers}
        if paper_id is not None and paper_id not in allowed:
            raise PermissionError("paper_id is outside the temporary workspace")
        if workspace.workspace_id not in self._retrievers:
            self.build(user_id=user_id, session_id=session_id, workspace_id=workspace_id)
        return self._retrievers[workspace.workspace_id].search(query, top_k, paper_id=paper_id, kind=kind)

    def evict(self, workspace_id: str) -> None:
        """Remove a workspace's cached and persisted index after its deletion."""

        self._retrievers.pop(workspace_id, None)
        self.remove_persisted(self.root, workspace_id)

    @staticmethod
    def remove_persisted(root_path: str | Path, workspace_id: str) -> None:
        # workspace_id was validated when the workspace was created; still
        # reject path-like values because this method performs deletion.
        if not workspace_id.startswith("tmpws_") or not workspace_id.replace("_", "").isalnum():
            raise ValueError("invalid temporary workspace_id")
        root = Path(root_path).resolve()
        target = (root / workspace_id).resolve()
        if not target.is_relative_to(root):
            raise ValueError("temporary index path escapes its root")
        if target.exists():
            shutil.rmtree(target)

    @staticmethod
    def _collection_name(workspace: TemporaryPaperWorkspace) -> str:
        return f"temporary_{workspace.workspace_id}"

    @staticmethod
    def _load_workspace_chunks(workspace: TemporaryPaperWorkspace) -> list[RetrievalChunk]:
        chunks: list[RetrievalChunk] = []
        for paper in workspace.papers:
            if not paper.chunks_path:
                raise ValueError(f"temporary paper has no chunks_path: {paper.paper_id}")
            source_chunks = chunks_from_bundles(load_chunk_bundles(paper.chunks_path))
            if any(chunk.paper_id != paper.paper_id for chunk in source_chunks):
                raise ValueError(f"chunks_path paper_id mismatch for temporary paper: {paper.paper_id}")
            chunks.extend(source_chunks)
        return chunks

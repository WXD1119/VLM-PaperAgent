from pathlib import Path

import pytest

from paper_agent.domain import ChunkBundle, ChunkKind, RetrievalChunk
from paper_agent.temporary_workspace import LocalTemporaryWorkspaceStore, TemporaryPaperRef, TemporaryWorkspaceIndexManager


class FakeEncoder:
    model_name = "fake"
    dimension = 2


class FakeDense:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks):
        self.chunks = list(chunks)

    def search(self, query, top_k=5, paper_id=None, kind=None):
        return []


def test_temporary_index_uses_only_workspace_papers(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    bundle = ChunkBundle(
        paper_id="paper-temp",
        parents=[],
        children=[RetrievalChunk(chunk_id="chunk-temp", parent_chunk_id="parent", paper_id="paper-temp", kind=ChunkKind.TEXT, content="temporary Q-Former evidence", element_ids=["e1"], pages=[1], section_path=["Method"])],
    )
    chunks_path.write_text(bundle.model_dump_json(), encoding="utf-8")
    store = LocalTemporaryWorkspaceStore(tmp_path / "workspaces")
    workspace = store.create(user_id="alice", session_id="session", papers=[TemporaryPaperRef(paper_id="paper-temp", chunks_path=str(chunks_path))])
    manager = TemporaryWorkspaceIndexManager(store, FakeEncoder(), root=tmp_path / "indexes", dense_factory=lambda *_args: FakeDense())
    assert manager.build(user_id="alice", session_id="session", workspace_id=workspace.workspace_id) == 1
    assert manager.search(user_id="alice", session_id="session", workspace_id=workspace.workspace_id, query="Q-Former")[0].paper_id == "paper-temp"
    with pytest.raises(PermissionError):
        manager.search(user_id="alice", session_id="session", workspace_id=workspace.workspace_id, query="Q-Former", paper_id="library-paper")

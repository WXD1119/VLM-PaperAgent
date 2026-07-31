from datetime import UTC, datetime, timedelta

import pytest

from paper_agent.temporary_workspace import LocalTemporaryWorkspaceStore, TemporaryPaperRef


def test_temporary_workspace_is_user_and_session_isolated(tmp_path):
    store = LocalTemporaryWorkspaceStore(tmp_path)
    workspace = store.create(
        user_id="alice",
        session_id="session_a",
        papers=[TemporaryPaperRef(paper_id="paper_new", ingestion_task_id="ing_1")],
    )

    loaded = store.load(user_id="alice", session_id="session_a", workspace_id=workspace.workspace_id)
    assert [paper.paper_id for paper in loaded.papers] == ["paper_new"]
    with pytest.raises(KeyError):
        store.load(user_id="bob", session_id="session_a", workspace_id=workspace.workspace_id)
    with pytest.raises(KeyError):
        store.load(user_id="alice", session_id="session_b", workspace_id=workspace.workspace_id)


def test_temporary_workspace_rejects_promoted_paper():
    with pytest.raises(ValueError, match="promoted"):
        TemporaryPaperRef(paper_id="paper_library", promoted_to_library=True)


def test_delete_is_owner_scoped_and_idempotent(tmp_path):
    store = LocalTemporaryWorkspaceStore(tmp_path)
    workspace = store.create(user_id="alice", session_id="s1", papers=[])

    assert not store.delete(user_id="bob", session_id="s1", workspace_id=workspace.workspace_id)
    assert store.delete(user_id="alice", session_id="s1", workspace_id=workspace.workspace_id)
    assert not store.delete(user_id="alice", session_id="s1", workspace_id=workspace.workspace_id)


def test_expired_workspace_is_removed_on_load_and_cleanup(tmp_path):
    store = LocalTemporaryWorkspaceStore(tmp_path)
    workspace = store.create(
        user_id="alice",
        session_id="s1",
        papers=[],
        ttl=timedelta(seconds=1),
    )
    future = workspace.expires_at + timedelta(seconds=1)

    assert store.cleanup_expired(now=future) == 1
    with pytest.raises(KeyError):
        store.load(user_id="alice", session_id="s1", workspace_id=workspace.workspace_id)


def test_cleanup_keeps_unexpired_workspaces(tmp_path):
    store = LocalTemporaryWorkspaceStore(tmp_path)
    workspace = store.create(user_id="alice", session_id="s1", papers=[], ttl=timedelta(hours=2))

    assert store.cleanup_expired(now=datetime.now(UTC)) == 0
    assert store.load(user_id="alice", session_id="s1", workspace_id=workspace.workspace_id).workspace_id == workspace.workspace_id


def test_temporary_workspace_calls_cleanup_hook_on_delete(tmp_path):
    removed: list[str] = []
    store = LocalTemporaryWorkspaceStore(tmp_path, on_delete=lambda workspace: removed.append(workspace.workspace_id))
    workspace = store.create(user_id="alice", session_id="session", papers=[])
    assert store.delete(user_id="alice", session_id="session", workspace_id=workspace.workspace_id) is True
    assert removed == [workspace.workspace_id]

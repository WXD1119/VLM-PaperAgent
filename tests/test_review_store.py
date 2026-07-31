import pytest

from paper_agent.agents import DeterministicReviewPlanner, ReviewExecutionResult
from paper_agent.reviews import LocalReviewArtifactStore, ReviewArtifact


def test_local_review_store_is_user_scoped(tmp_path):
    store = LocalReviewArtifactStore(tmp_path)
    artifact = store.create(
        ReviewArtifact(
            user_id="alice",
            execution=ReviewExecutionResult(
                plan=DeterministicReviewPlanner().plan("Topic", ["paper-a", "paper-b"]),
            ),
        )
    )
    assert store.load("alice", artifact.review_id).review_id == artifact.review_id
    assert [item.review_id for item in store.list("alice")] == [artifact.review_id]
    with pytest.raises(KeyError):
        store.load("bob", artifact.review_id)

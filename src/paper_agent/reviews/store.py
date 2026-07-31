from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paper_agent.reviews.models import ReviewArtifact


class ReviewArtifactStore(Protocol):
    def create(self, artifact: ReviewArtifact) -> ReviewArtifact: ...

    def load(self, user_id: str, review_id: str) -> ReviewArtifact: ...

    def list(self, user_id: str, limit: int = 20) -> list[ReviewArtifact]: ...


class LocalReviewArtifactStore:
    """Local development persistence; each review remains a user-owned JSON artifact."""

    def __init__(self, root: str | Path = "artifacts/reviews") -> None:
        self.root = Path(root)

    def create(self, artifact: ReviewArtifact) -> ReviewArtifact:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(artifact.review_id).write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        return artifact

    def load(self, user_id: str, review_id: str) -> ReviewArtifact:
        artifact = ReviewArtifact.model_validate_json(self._path(review_id).read_text(encoding="utf-8"))
        if artifact.user_id != user_id:
            # Do not disclose the existence of another user's artifact.
            raise KeyError(review_id)
        return artifact

    def list(self, user_id: str, limit: int = 20) -> list[ReviewArtifact]:
        if not self.root.exists():
            return []
        artifacts = [ReviewArtifact.model_validate_json(path.read_text(encoding="utf-8")) for path in self.root.glob("review_*.json")]
        return sorted(
            (item for item in artifacts if item.user_id == user_id),
            key=lambda item: item.updated_at,
            reverse=True,
        )[:limit]

    def _path(self, review_id: str) -> Path:
        if not review_id.startswith("review_") or not review_id.replace("_", "").isalnum():
            raise ValueError("invalid review_id")
        return self.root / f"{review_id}.json"

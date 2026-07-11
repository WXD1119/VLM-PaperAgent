from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Stable user-level preferences and environment defaults.

    This is agent memory, not paper knowledge. It can store operational preferences such
    as the default server path or preferred language, but it must not be merged into the
    paper knowledge graph.
    """

    preferred_language: str | None = None
    default_workspace_id: str | None = None
    server_project_path: str | None = None
    conda_env: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)


class UserProfileStore:
    def __init__(self, path: str | Path = "artifacts/memory/user_profile.json") -> None:
        self.path = Path(path)

    def load(self) -> UserProfile:
        if not self.path.exists():
            return UserProfile()
        return UserProfile.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, profile: UserProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def set(self, key: str, value: Any) -> UserProfile:
        profile = self.load()
        if key in UserProfile.model_fields:
            profile = profile.model_copy(update={key: value})
        else:
            preferences = dict(profile.preferences)
            preferences[key] = value
            profile = profile.model_copy(update={"preferences": preferences})
        self.save(profile)
        return profile

"""Production storage adapters for the three-layer memory boundary.

Redis holds expiring session state and a bounded recent conversation window.
MySQL holds durable user-owned memory. Neo4j intentionally remains the Paper KG
backend and is never used for user conversation or profile data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from paper_agent.memory.episodic import Episode
from paper_agent.memory.profile import UserProfile
from paper_agent.memory.session import SessionState
from paper_agent.memory.summary import ConversationSummary, ConversationTurn


class RedisSessionMemoryStore:
    def __init__(
        self,
        redis_url: str,
        *,
        user_id: str,
        session_id: str,
        ttl_seconds: int = 86_400,
        client=None,
    ) -> None:
        if not user_id or not session_id:
            raise ValueError("user_id and session_id are required for Redis session memory")
        if ttl_seconds < 60:
            raise ValueError("Redis session TTL must be at least 60 seconds")
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError("Install storage support: pip install -e '.[storage]'") from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.key = f"paper_agent:session:{_key_part(user_id)}:{_key_part(session_id)}"

    def load(self) -> SessionState:
        value = self.client.get(self.key)
        if not value:
            return SessionState()
        return SessionState.model_validate_json(value)

    def save(self, state: SessionState) -> None:
        self.client.set(self.key, state.model_dump_json(), ex=self.ttl_seconds)

    def update(self, **changes: Any) -> SessionState:
        state = self.load()
        updated = state.model_copy(update={key: value for key, value in changes.items() if value is not None})
        self.save(updated)
        return updated

    def clear(self) -> None:
        self.client.delete(self.key)


class RedisConversationWindowStore:
    """会过期的原始对话窗口，不属于长期记忆。"""

    def __init__(
        self,
        redis_url: str,
        *,
        user_id: str,
        session_id: str,
        max_turns: int = 12,
        ttl_seconds: int = 86_400,
        client=None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.session = RedisSessionMemoryStore(
            redis_url,
            user_id=user_id,
            session_id=session_id,
            ttl_seconds=ttl_seconds,
            client=client,
        )
        self.client = self.session.client
        self.max_turns = max_turns
        self.key = self.session.key.replace("paper_agent:session:", "paper_agent:turns:")

    def append(self, turn: ConversationTurn) -> ConversationTurn:
        self.client.rpush(self.key, turn.model_dump_json())
        self.client.ltrim(self.key, -self.max_turns, -1)
        self.client.expire(self.key, self.session.ttl_seconds)
        return turn

    def list(self) -> list[ConversationTurn]:
        values = self.client.lrange(self.key, 0, -1)
        return [ConversationTurn.model_validate_json(value) for value in values]

    def clear(self) -> None:
        self.client.delete(self.key)


class MySQLLongTermMemoryStore:
    """使用 SQLAlchemy 和 MySQL 兼容 Schema 的持久用户记忆仓储。"""

    def __init__(self, database_url: str, *, engine=None) -> None:
        if engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:
                raise RuntimeError("Install storage support: pip install -e '.[storage]'") from exc
            engine = create_engine(database_url, pool_pre_ping=True)
        self.engine = engine
        self._define_schema()

    def _define_schema(self) -> None:
        try:
            from sqlalchemy import Column, MetaData, String, Table, Text
        except ImportError as exc:
            raise RuntimeError("Install storage support: pip install -e '.[storage]'") from exc
        metadata = MetaData()
        self.profiles = Table(
            "agent_user_profiles",
            metadata,
            Column("user_id", String(128), primary_key=True),
            Column("profile_json", Text, nullable=False),
            Column("updated_at", String(40), nullable=False),
        )
        self.episodes = Table(
            "agent_episodes",
            metadata,
            Column("event_id", String(64), primary_key=True),
            Column("user_id", String(128), nullable=False, index=True),
            Column("session_id", String(128), nullable=True, index=True),
            Column("timestamp", String(40), nullable=False),
            Column("event_type", String(128), nullable=False),
            Column("summary", Text, nullable=False),
            Column("payload_json", Text, nullable=False),
        )
        self.summaries = Table(
            "agent_summaries",
            metadata,
            Column("summary_id", String(64), primary_key=True),
            Column("user_id", String(128), nullable=False, index=True),
            Column("session_id", String(128), nullable=True, index=True),
            Column("created_at", String(40), nullable=False),
            Column("summary_json", Text, nullable=False),
        )
        self.turns = Table(
            "agent_conversation_turns",
            metadata,
            Column("turn_id", String(64), primary_key=True),
            Column("user_id", String(128), nullable=False, index=True),
            Column("session_id", String(128), nullable=False, index=True),
            Column("timestamp", String(40), nullable=False),
            Column("role", String(32), nullable=False),
            Column("content", Text, nullable=False),
            Column("metadata_json", Text, nullable=False),
        )
        metadata.create_all(self.engine)

    def load_profile(self, user_id: str) -> UserProfile:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.profiles.c.profile_json).where(self.profiles.c.user_id == user_id)
            ).first()
        return UserProfile.model_validate_json(row[0]) if row else UserProfile()

    def save_profile(self, user_id: str, profile: UserProfile) -> UserProfile:
        from sqlalchemy import delete, insert

        now = datetime.now(UTC).isoformat()
        with self.engine.begin() as connection:
            connection.execute(delete(self.profiles).where(self.profiles.c.user_id == user_id))
            connection.execute(
                insert(self.profiles).values(
                    user_id=user_id,
                    profile_json=profile.model_dump_json(),
                    updated_at=now,
                )
            )
        return profile

    def append_episode(self, user_id: str, episode: Episode, *, session_id: str | None = None) -> Episode:
        from sqlalchemy import insert

        with self.engine.begin() as connection:
            connection.execute(
                insert(self.episodes).values(
                    event_id=episode.event_id,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=episode.timestamp,
                    event_type=episode.event_type,
                    summary=episode.summary,
                    payload_json=json.dumps(episode.payload, ensure_ascii=False),
                )
            )
        return episode

    def list_episodes(self, user_id: str, *, session_id: str | None = None, limit: int = 100) -> list[Episode]:
        from sqlalchemy import select

        statement = select(self.episodes).where(self.episodes.c.user_id == user_id)
        if session_id is not None:
            statement = statement.where(self.episodes.c.session_id == session_id)
        statement = statement.order_by(self.episodes.c.timestamp.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = list(connection.execute(statement).mappings())
        return [
            Episode(
                event_id=row["event_id"],
                timestamp=row["timestamp"],
                event_type=row["event_type"],
                summary=row["summary"],
                payload=json.loads(row["payload_json"]),
            )
            for row in reversed(rows)
        ]

    def append_summary(
        self, user_id: str, summary: ConversationSummary, *, session_id: str | None = None
    ) -> ConversationSummary:
        from sqlalchemy import insert

        with self.engine.begin() as connection:
            connection.execute(
                insert(self.summaries).values(
                    summary_id=summary.summary_id,
                    user_id=user_id,
                    session_id=session_id,
                    created_at=summary.created_at,
                    summary_json=summary.model_dump_json(),
                )
            )
        return summary

    def list_summaries(
        self, user_id: str, *, session_id: str | None = None, limit: int = 100
    ) -> list[ConversationSummary]:
        from sqlalchemy import select

        statement = select(self.summaries).where(self.summaries.c.user_id == user_id)
        if session_id is not None:
            statement = statement.where(self.summaries.c.session_id == session_id)
        statement = statement.order_by(self.summaries.c.created_at.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = list(connection.execute(statement).mappings())
        return [ConversationSummary.model_validate_json(row["summary_json"]) for row in reversed(rows)]

    def append_turn(self, user_id: str, session_id: str, turn: ConversationTurn) -> ConversationTurn:
        from sqlalchemy import insert

        with self.engine.begin() as connection:
            connection.execute(
                insert(self.turns).values(
                    turn_id=turn.turn_id,
                    user_id=user_id,
                    session_id=session_id,
                    timestamp=turn.timestamp,
                    role=turn.role,
                    content=turn.content,
                    metadata_json=json.dumps(turn.metadata, ensure_ascii=False),
                )
            )
        return turn


class MySQLUserProfileStore:
    """与既有画像存储契约兼容的用户级适配器。"""

    def __init__(self, store: MySQLLongTermMemoryStore, user_id: str) -> None:
        self.store = store
        self.user_id = user_id

    def load(self) -> UserProfile:
        return self.store.load_profile(self.user_id)

    def save(self, profile: UserProfile) -> None:
        self.store.save_profile(self.user_id, profile)

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


class MySQLEpisodicMemoryStore:
    """与事件记忆方法兼容的用户/会话级适配器。"""

    def __init__(
        self, store: MySQLLongTermMemoryStore, user_id: str, *, session_id: str | None = None
    ) -> None:
        self.store = store
        self.user_id = user_id
        self.session_id = session_id

    def append(self, episode: Episode) -> Episode:
        return self.store.append_episode(self.user_id, episode, session_id=self.session_id)

    def log(
        self, event_type: str, summary: str, payload: dict[str, Any] | None = None
    ) -> Episode:
        return self.append(Episode(event_type=event_type, summary=summary, payload=payload or {}))

    def list(self, limit: int | None = None) -> list[Episode]:
        return self.store.list_episodes(
            self.user_id, session_id=self.session_id, limit=limit or 100
        )


def _key_part(value: str) -> str:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("memory key contains unsupported characters")
    return value

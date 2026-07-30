import pytest

pytest.importorskip("sqlalchemy")

from paper_agent.memory import (
    ConversationTurn,
    Episode,
    MySQLLongTermMemoryStore,
    RedisConversationWindowStore,
    RedisSessionMemoryStore,
    SessionState,
    UserProfile,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)
        self.lists.pop(key, None)

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        self.lists[key] = self.lists.get(key, [])[start : end + 1 if end != -1 else None]

    def expire(self, key, seconds):
        return True

    def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        return values[start : end + 1 if end != -1 else None]


def test_redis_stores_session_and_bounded_recent_turns():
    client = FakeRedis()
    session = RedisSessionMemoryStore(
        "redis://unused", user_id="wxd", session_id="s1", client=client
    )
    session.save(SessionState(current_paper_id="paper-1"))
    assert session.load().current_paper_id == "paper-1"

    turns = RedisConversationWindowStore(
        "redis://unused", user_id="wxd", session_id="s1", max_turns=2, client=client
    )
    turns.append(ConversationTurn(role="user", content="one"))
    turns.append(ConversationTurn(role="assistant", content="two"))
    turns.append(ConversationTurn(role="user", content="three"))
    assert [turn.content for turn in turns.list()] == ["two", "three"]


def test_mysql_long_term_store_keeps_profile_and_episodes():
    store = MySQLLongTermMemoryStore("sqlite+pysqlite:///:memory:")
    store.save_profile("wxd", UserProfile(preferred_language="zh"))
    assert store.load_profile("wxd").preferred_language == "zh"

    store.append_episode("wxd", Episode(event_type="decision", summary="keep Paper KG clean"))
    assert store.list_episodes("wxd")[0].summary == "keep Paper KG clean"

"""Redis、MySQL、Neo4j 的最小只读健康检查。"""

from __future__ import annotations

from pydantic import BaseModel


class StorageCheck(BaseModel):
    name: str
    status: str
    detail: str | None = None


class StorageHealthReport(BaseModel):
    checks: list[StorageCheck]

    @property
    def ready(self) -> bool:
        return all(check.status in {"ok", "not_configured"} for check in self.checks)


def check_storage_health(
    *,
    redis_url: str | None = None,
    mysql_url: str | None = None,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> StorageHealthReport:
    """只执行 ping/SELECT 1/连接校验，不读取任何用户或论文数据。"""

    return StorageHealthReport(
        checks=[
            _check_redis(redis_url),
            _check_mysql(mysql_url),
            _check_neo4j(neo4j_uri, neo4j_user, neo4j_password),
        ]
    )


def _check_redis(redis_url: str | None) -> StorageCheck:
    if not redis_url:
        return StorageCheck(name="redis", status="not_configured")
    try:
        import redis

        client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        return StorageCheck(name="redis", status="ok")
    except Exception as exc:
        return StorageCheck(name="redis", status="error", detail=_safe_detail(exc))


def _check_mysql(mysql_url: str | None) -> StorageCheck:
    if not mysql_url:
        return StorageCheck(name="mysql", status="not_configured")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(mysql_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
        return StorageCheck(name="mysql", status="ok")
    except Exception as exc:
        return StorageCheck(name="mysql", status="error", detail=_safe_detail(exc))


def _check_neo4j(uri: str | None, user: str | None, password: str | None) -> StorageCheck:
    if not uri:
        return StorageCheck(name="neo4j", status="not_configured")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user or "neo4j", password or ""), connection_timeout=2)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return StorageCheck(name="neo4j", status="ok")
    except Exception as exc:
        return StorageCheck(name="neo4j", status="error", detail=_safe_detail(exc))


def _safe_detail(exc: Exception) -> str:
    """错误信息不回显连接串、密码或完整服务端异常。"""

    return type(exc).__name__

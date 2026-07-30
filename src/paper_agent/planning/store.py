"""研究计划仓储：离线 JSON 回退与 MySQL 持久化实现。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from paper_agent.planning.models import ResearchPlan


class ResearchPlanStore(Protocol):
    def create(self, plan: ResearchPlan) -> ResearchPlan: ...

    def load(self, user_id: str, plan_id: str) -> ResearchPlan: ...

    def list(self, user_id: str, limit: int = 20) -> list[ResearchPlan]: ...

    def save(self, plan: ResearchPlan) -> ResearchPlan: ...


class LocalResearchPlanStore:
    """本地开发回退仓储；每个计划单独保存，便于人工检查。"""

    def __init__(self, root: str | Path = "artifacts/research_plans") -> None:
        self.root = Path(root)

    def create(self, plan: ResearchPlan) -> ResearchPlan:
        return self.save(plan)

    def load(self, user_id: str, plan_id: str) -> ResearchPlan:
        plan = ResearchPlan.model_validate_json(self._path(plan_id).read_text(encoding="utf-8"))
        if plan.user_id != user_id:
            raise KeyError(plan_id)
        return plan

    def list(self, user_id: str, limit: int = 20) -> list[ResearchPlan]:
        if not self.root.exists():
            return []
        plans = [
            ResearchPlan.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*.json")
        ]
        return sorted((plan for plan in plans if plan.user_id == user_id), key=lambda plan: plan.updated_at, reverse=True)[:limit]

    def save(self, plan: ResearchPlan) -> ResearchPlan:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(plan.plan_id).write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return plan

    def _path(self, plan_id: str) -> Path:
        if not plan_id.startswith("plan_") or not plan_id.replace("_", "").isalnum():
            raise ValueError("非法 plan_id")
        return self.root / f"{plan_id}.json"


class MySQLResearchPlanStore:
    """MySQL 计划仓储；任务与结果作为用户记忆，不写入论文图谱。"""

    def __init__(self, database_url: str) -> None:
        from sqlalchemy import Column, MetaData, String, Table, Text, create_engine

        self.engine = create_engine(database_url, pool_pre_ping=True)
        metadata = MetaData()
        self.table = Table(
            "agent_research_plans",
            metadata,
            Column("plan_id", String(64), primary_key=True),
            Column("user_id", String(128), nullable=False, index=True),
            Column("updated_at", String(40), nullable=False, index=True),
            Column("plan_json", Text, nullable=False),
        )
        metadata.create_all(self.engine)

    def create(self, plan: ResearchPlan) -> ResearchPlan:
        return self.save(plan)

    def load(self, user_id: str, plan_id: str) -> ResearchPlan:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.table.c.plan_json).where(self.table.c.plan_id == plan_id, self.table.c.user_id == user_id)
            ).first()
        if not row:
            raise KeyError(plan_id)
        return ResearchPlan.model_validate_json(row[0])

    def list(self, user_id: str, limit: int = 20) -> list[ResearchPlan]:
        from sqlalchemy import select

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(self.table.c.plan_json).where(self.table.c.user_id == user_id).order_by(self.table.c.updated_at.desc()).limit(limit)
            ).all()
        return [ResearchPlan.model_validate_json(row[0]) for row in rows]

    def save(self, plan: ResearchPlan) -> ResearchPlan:
        from sqlalchemy import delete, insert

        with self.engine.begin() as connection:
            connection.execute(delete(self.table).where(self.table.c.plan_id == plan.plan_id))
            connection.execute(insert(self.table).values(plan_id=plan.plan_id, user_id=plan.user_id, updated_at=plan.updated_at, plan_json=plan.model_dump_json()))
        return plan

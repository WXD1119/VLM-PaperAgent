"""用户可见、可编辑且受约束的论文调研计划模型。"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ResearchTaskType(StrEnum):
    PAPER_QA = "paper_qa"
    SYNTHESIZE = "synthesize"


class ResearchTaskStatus(StrEnum):
    TODO = "todo"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_CONFIRMATION = "waiting_confirmation"


class ResearchPlanStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_CONFIRMATION = "waiting_confirmation"


class ResearchTask(BaseModel):
    """执行器允许处理的一项白名单研究子任务。"""

    task_id: str = Field(default_factory=lambda: f"rt_{uuid4().hex[:16]}")
    title: str
    task_type: ResearchTaskType
    status: ResearchTaskStatus = ResearchTaskStatus.TODO
    paper_id: str | None = None
    question: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    result_summary: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ResearchTask":
        if self.task_type == ResearchTaskType.PAPER_QA and (not self.paper_id or not self.question):
            raise ValueError("paper_qa 任务必须包含 paper_id 和 question")
        return self


class ResearchPlan(BaseModel):
    """复杂调研的用户可控执行计划，不是模型的隐藏思维链。"""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:16]}")
    user_id: str
    goal: str = Field(min_length=1, max_length=2000)
    status: ResearchPlanStatus = ResearchPlanStatus.DRAFT
    tasks: list[ResearchTask] = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    final_report: str | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> "ResearchPlan":
        task_ids = {task.task_id for task in self.tasks}
        for task in self.tasks:
            unknown = set(task.depends_on) - task_ids
            if unknown:
                raise ValueError(f"任务 {task.task_id} 依赖不存在：{sorted(unknown)}")
        return self

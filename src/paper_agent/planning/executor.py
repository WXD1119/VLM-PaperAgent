"""受限调研计划执行器。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from paper_agent.planning.models import (
    ResearchPlan,
    ResearchPlanStatus,
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
)
from paper_agent.planning.store import ResearchPlanStore


class PaperQAResult(BaseModel):
    """单篇论文问答链交给规划执行器的最小、脱敏结果。"""

    answer: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    citation_valid: bool
    abstained: bool


PaperQACallable = Callable[[str, str], PaperQAResult]


class ResearchPlanExecutor:
    """只执行白名单任务类型，并将每次执行结果持久化到用户侧计划存储。"""

    def __init__(self, store: ResearchPlanStore, ask_paper: PaperQACallable) -> None:
        self.store = store
        self.ask_paper = ask_paper

    def execute(self, *, user_id: str, plan_id: str, confirmed: bool) -> ResearchPlan:
        """执行草案；未确认的计划绝不触发模型、检索或图谱写入。"""

        plan = self.store.load(user_id, plan_id)
        if not confirmed:
            plan.status = ResearchPlanStatus.WAITING_CONFIRMATION
            plan.updated_at = _now()
            return self.store.save(plan)
        if plan.status == ResearchPlanStatus.COMPLETED:
            return plan

        plan.status = ResearchPlanStatus.RUNNING
        plan.updated_at = _now()
        self.store.save(plan)
        for task in plan.tasks:
            if task.task_type != ResearchTaskType.PAPER_QA or task.status == ResearchTaskStatus.COMPLETED:
                continue
            self._execute_paper_qa(plan, task)
            self.store.save(plan)

        if any(task.status == ResearchTaskStatus.FAILED for task in plan.tasks):
            plan.status = ResearchPlanStatus.FAILED
            plan.updated_at = _now()
            return self.store.save(plan)

        synthesis = next(task for task in plan.tasks if task.task_type == ResearchTaskType.SYNTHESIZE)
        if synthesis.status != ResearchTaskStatus.COMPLETED:
            self._synthesize(plan, synthesis)
        plan.status = ResearchPlanStatus.COMPLETED
        plan.updated_at = _now()
        return self.store.save(plan)

    def _execute_paper_qa(self, plan: ResearchPlan, task: ResearchTask) -> None:
        task.status = ResearchTaskStatus.RUNNING
        task.error = None
        try:
            result = self.ask_paper(task.question or "", task.paper_id or "")
            task.result_summary = result.answer
            task.evidence_chunk_ids = result.evidence_chunk_ids
            task.trace_id = result.trace_id
            if not result.citation_valid:
                raise ValueError("子任务引用完整性校验未通过")
            task.status = ResearchTaskStatus.COMPLETED
        except Exception as exc:
            # 计划层只保存受限错误摘要，避免把模型输出或隐私内容扩散到长期存储。
            task.status = ResearchTaskStatus.FAILED
            task.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            plan.updated_at = _now()

    def _synthesize(self, plan: ResearchPlan, task: ResearchTask) -> None:
        sources = [item for item in plan.tasks if item.task_type == ResearchTaskType.PAPER_QA]
        if any(item.status != ResearchTaskStatus.COMPLETED for item in sources):
            task.status = ResearchTaskStatus.FAILED
            task.error = "依赖的论文问答子任务尚未全部完成"
            return
        # 不再调用自由生成模型做无约束“综合”；报告仅拼接各子任务的可追溯结论。
        lines = [f"调研目标：{plan.goal}", "", "## 分论文结论"]
        evidence_ids: list[str] = []
        for item in sources:
            lines.extend(
                [
                    f"### {item.paper_id}",
                    item.result_summary or "该论文子任务未产出结论。",
                    f"证据 Chunk：{', '.join(item.evidence_chunk_ids) or '无'}",
                    "",
                ]
            )
            evidence_ids.extend(item.evidence_chunk_ids)
        lines.extend(
            [
                "## 使用说明",
                "以上内容按论文分别保留；跨论文比较请回看每一项的证据 Chunk，而不是将本汇总视为新的论文事实。",
            ]
        )
        task.status = ResearchTaskStatus.COMPLETED
        task.result_summary = "\n".join(lines)
        task.evidence_chunk_ids = list(dict.fromkeys(evidence_ids))
        plan.final_report = task.result_summary


def _now() -> str:
    return datetime.now(UTC).isoformat()

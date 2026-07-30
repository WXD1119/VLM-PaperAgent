"""研究计划生成器：输出受 Schema 限制的任务，而非开放式思维链。"""

from __future__ import annotations

from paper_agent.planning.models import ResearchPlan, ResearchTask, ResearchTaskType


class DeterministicResearchPlanner:
    """根据目标和明确论文范围创建可编辑的对比调研计划。"""

    def create_comparison_plan(self, *, user_id: str, goal: str, paper_ids: list[str]) -> ResearchPlan:
        if len(paper_ids) < 2:
            raise ValueError("对比调研至少需要两篇论文")
        normalized = list(dict.fromkeys(paper_id.strip() for paper_id in paper_ids if paper_id.strip()))
        if len(normalized) < 2:
            raise ValueError("至少需要两个不同的 paper_id")
        tasks = [
            ResearchTask(
                title=f"提取 {paper_id} 的可追溯结论",
                task_type=ResearchTaskType.PAPER_QA,
                paper_id=paper_id,
                question=f"围绕以下调研目标，仅根据本文总结方法、训练方式、实验结论与局限：{goal}",
            )
            for paper_id in normalized
        ]
        tasks.append(
            ResearchTask(
                title="汇总跨论文对比结论",
                task_type=ResearchTaskType.SYNTHESIZE,
                depends_on=[task.task_id for task in tasks],
            )
        )
        return ResearchPlan(user_id=user_id, goal=goal, tasks=tasks)

from paper_agent.planning import (
    DeterministicResearchPlanner,
    LocalResearchPlanStore,
    PaperQAResult,
    ResearchPlanExecutor,
    ResearchPlanStatus,
    ResearchTaskStatus,
)


def test_comparison_plan_is_user_visible_and_has_explicit_dependencies():
    plan = DeterministicResearchPlanner().create_comparison_plan(
        user_id="wxd",
        goal="比较视觉与语言连接方式",
        paper_ids=["paper-a", "paper-b"],
    )

    assert plan.status == ResearchPlanStatus.DRAFT
    assert len(plan.tasks) == 3
    assert {task.paper_id for task in plan.tasks[:-1]} == {"paper-a", "paper-b"}
    assert set(plan.tasks[-1].depends_on) == {task.task_id for task in plan.tasks[:-1]}


def test_plan_requires_confirmation_before_calling_paper_qa(tmp_path):
    store = LocalResearchPlanStore(tmp_path)
    plan = store.create(
        DeterministicResearchPlanner().create_comparison_plan(
            user_id="wxd", goal="比较两篇论文", paper_ids=["paper-a", "paper-b"]
        )
    )
    calls: list[tuple[str, str]] = []

    def ask_paper(question: str, paper_id: str) -> PaperQAResult:
        calls.append((question, paper_id))
        return PaperQAResult(
            answer=f"{paper_id} 的可追溯结论",
            evidence_chunk_ids=[f"{paper_id}-chunk"],
            trace_id=f"trace-{paper_id}",
            citation_valid=True,
            abstained=False,
        )

    executor = ResearchPlanExecutor(store, ask_paper)
    waiting = executor.execute(user_id="wxd", plan_id=plan.plan_id, confirmed=False)
    assert waiting.status == ResearchPlanStatus.WAITING_CONFIRMATION
    assert calls == []

    completed = executor.execute(user_id="wxd", plan_id=plan.plan_id, confirmed=True)
    assert completed.status == ResearchPlanStatus.COMPLETED
    assert len(calls) == 2
    assert completed.final_report is not None
    assert "paper-a-chunk" in completed.final_report
    assert all(task.status == ResearchTaskStatus.COMPLETED for task in completed.tasks)


def test_plan_store_is_scoped_by_user(tmp_path):
    store = LocalResearchPlanStore(tmp_path)
    plan = store.create(
        DeterministicResearchPlanner().create_comparison_plan(
            user_id="wxd", goal="比较两篇论文", paper_ids=["paper-a", "paper-b"]
        )
    )

    assert store.list("other-user") == []
    try:
        store.load("other-user", plan.plan_id)
    except KeyError:
        pass
    else:
        raise AssertionError("不同用户不应读取到该调研计划")

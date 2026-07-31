from paper_agent.agents import (
    ComparisonBudget,
    DeterministicComparisonPlanner,
    ReadingCompareServices,
    build_reading_compare_graph,
)
from paper_agent.planning import PaperQAResult


def test_comparison_planner_caps_papers_and_dimensions():
    plan = DeterministicComparisonPlanner().plan(
        "Compare architecture and experiment results",
        ["paper-a", "paper-b"],
        ComparisonBudget(max_papers=2, max_dimensions=2),
    )
    assert plan.paper_ids == ["paper-a", "paper-b"]
    assert len(plan.dimensions) == 2


def test_reading_compare_graph_builds_citable_matrix():
    import pytest

    pytest.importorskip("langgraph")
    calls: list[tuple[str, str]] = []

    def ask_paper(question: str, paper_id: str) -> PaperQAResult:
        calls.append((question, paper_id))
        return PaperQAResult(answer=f"{paper_id} answer", evidence_chunk_ids=[f"{paper_id}-chunk"], citation_valid=True, abstained=False)

    graph = build_reading_compare_graph(ReadingCompareServices(ask_paper=ask_paper))
    state = graph.invoke({"query": "Compare architecture", "paper_ids": ["paper-a", "paper-b"], "budget": ComparisonBudget(max_dimensions=1)})
    matrix = state["matrix"]
    assert len(calls) == 2
    assert len(matrix.cells) == 2
    assert all(cell.citation_valid for cell in matrix.cells)
    assert matrix.cells[0].evidence_chunk_ids == ["paper-a-chunk"]

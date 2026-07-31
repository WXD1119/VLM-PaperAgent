import pytest

from paper_agent.agents import DeterministicReviewPlanner, ReviewBudget, ReviewTemplate


def test_review_planner_requires_selected_corpus_and_caps_scope():
    plan = DeterministicReviewPlanner().plan(
        "Vision-language model connectors",
        ["paper-a", "paper-b"],
        ReviewTemplate.CONFERENCE_STYLE,
        ReviewBudget(max_papers=2, max_sections=3),
    )
    assert len(plan.sections) == 3
    assert plan.paper_ids == ["paper-a", "paper-b"]


from paper_agent.agents import ReviewSectionDraft, ReviewWriterServices, build_review_writer_graph
from paper_agent.planning import PaperQAResult


def test_review_graph_waits_for_explicit_confirmation_before_evidence_calls():
    pytest.importorskip("langgraph")
    calls: list[str] = []
    graph = build_review_writer_graph(
        ReviewWriterServices(
            ask_paper=lambda question, paper_id: calls.append(paper_id) or PaperQAResult(answer="e", citation_valid=True, abstained=False),
            write_section=lambda section, evidence: ReviewSectionDraft(section_id=section.section_id, title=section.title, content="draft", citation_valid=True),
        )
    )
    state = graph.invoke({"topic": "Topic", "paper_ids": ["paper-a", "paper-b"], "confirmed": False})
    assert state["waiting_confirmation"] is True
    assert calls == []

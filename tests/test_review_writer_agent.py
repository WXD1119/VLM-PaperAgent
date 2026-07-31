import pytest

from paper_agent.agents import (
    DeterministicReviewPlanner,
    EvidenceBoundSectionWriter,
    ReviewBudget,
    ReviewClaim,
    ReviewEvidence,
    ReviewSectionDraft,
    ReviewSectionPlan,
    ReviewTemplate,
)


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


class FakeStructuredClient:
    def __init__(self, response):
        self.response = response

    def generate_structured(self, _prompt, _model):
        return self.response


def test_llm_review_writer_preserves_only_allowed_evidence_ids():
    section = ReviewSectionPlan(section_id="taxonomy", title="Taxonomy", objective="Compare", evidence_dimensions=["method"])
    evidence = [ReviewEvidence(section_id="taxonomy", paper_id="paper-a", summary="A uses queries.", evidence_chunk_ids=["chunk-a"], citation_valid=True, abstained=False)]
    returned = ReviewSectionDraft(
        section_id="untrusted", title="untrusted", content="A uses queries.", evidence_chunk_ids=["chunk-a"], citation_valid=False,
        claims=[ReviewClaim(text="A uses queries.", evidence_chunk_ids=["chunk-a"])],
    )
    draft = EvidenceBoundSectionWriter(FakeStructuredClient(returned)).write(section, evidence)
    assert draft.section_id == "taxonomy"
    assert draft.citation_valid is True
    assert draft.evidence_chunk_ids == ["chunk-a"]


def test_llm_review_writer_falls_back_when_model_cites_unknown_chunk():
    section = ReviewSectionPlan(section_id="taxonomy", title="Taxonomy", objective="Compare", evidence_dimensions=["method"])
    evidence = [ReviewEvidence(section_id="taxonomy", paper_id="paper-a", summary="A uses queries.", evidence_chunk_ids=["chunk-a"], citation_valid=True, abstained=False)]
    invalid = ReviewSectionDraft(section_id="taxonomy", title="Taxonomy", content="Unsupported", evidence_chunk_ids=["other"], citation_valid=True)
    draft = EvidenceBoundSectionWriter(FakeStructuredClient(invalid)).write(section, evidence)
    assert draft.evidence_chunk_ids == ["chunk-a"]
    assert "A uses queries" in draft.content

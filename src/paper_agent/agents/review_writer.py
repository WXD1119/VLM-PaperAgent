"""Bounded Plan-and-Execute + Reflection workflow for literature reviews."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field

from paper_agent.planning import PaperQAResult


class ReviewTemplate(StrEnum):
    RELATED_WORK = "related_work"
    SURVEY = "survey"
    CONFERENCE_STYLE = "conference_style"


class ReviewBudget(BaseModel):
    max_papers: int = Field(default=12, ge=2, le=30)
    max_sections: int = Field(default=6, ge=2, le=10)
    max_evidence_tasks: int = Field(default=24, ge=2, le=80)
    max_reflection_attempts: int = Field(default=2, ge=0, le=3)


class ReviewSectionPlan(BaseModel):
    section_id: str
    title: str
    objective: str
    evidence_dimensions: list[str] = Field(min_length=1)


class LiteratureReviewPlan(BaseModel):
    topic: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=2)
    template: ReviewTemplate
    sections: list[ReviewSectionPlan] = Field(min_length=2)
    budget: ReviewBudget = Field(default_factory=ReviewBudget)


class ReviewEvidence(BaseModel):
    section_id: str
    paper_id: str
    summary: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_valid: bool
    abstained: bool


class ReviewSectionDraft(BaseModel):
    section_id: str
    title: str
    content: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_valid: bool


class ReflectionAction(StrEnum):
    PASS = "pass"
    RETRIEVE_MORE = "retrieve_more"
    REWRITE_SECTION = "rewrite_section"
    ASK_USER_CLARIFICATION = "ask_user_clarification"
    MARK_INSUFFICIENT_EVIDENCE = "mark_insufficient_evidence"


class ReviewReflection(BaseModel):
    action: ReflectionAction
    affected_section_ids: list[str] = Field(default_factory=list)
    reasoning: str


class DeterministicReviewPlanner:
    """Creates a review plan that is intentionally small enough to inspect."""

    def plan(
        self,
        topic: str,
        paper_ids: list[str],
        template: ReviewTemplate = ReviewTemplate.CONFERENCE_STYLE,
        budget: ReviewBudget | None = None,
    ) -> LiteratureReviewPlan:
        limits = budget or ReviewBudget()
        normalized = list(dict.fromkeys(item.strip() for item in paper_ids if item.strip()))
        if len(normalized) < 2:
            raise ValueError("literature review requires at least two selected papers")
        if len(normalized) > limits.max_papers:
            raise ValueError(f"review exceeds max_papers={limits.max_papers}")
        sections = self._sections(template)[: limits.max_sections]
        return LiteratureReviewPlan(topic=topic, paper_ids=normalized, template=template, sections=sections, budget=limits)

    @staticmethod
    def _sections(template: ReviewTemplate) -> list[ReviewSectionPlan]:
        if template == ReviewTemplate.RELATED_WORK:
            return [
                ReviewSectionPlan(section_id="positioning", title="Research positioning", objective="Define the problem and scope.", evidence_dimensions=["problem and setting"]),
                ReviewSectionPlan(section_id="taxonomy", title="Method taxonomy", objective="Group methods by technical approach.", evidence_dimensions=["method architecture", "training objective"]),
                ReviewSectionPlan(section_id="gaps", title="Limitations and gaps", objective="Report evidenced limitations without inventing novelty claims.", evidence_dimensions=["limitations", "experimental evidence"]),
            ]
        return [
            ReviewSectionPlan(section_id="introduction", title="Introduction and scope", objective="Define scope and corpus boundaries.", evidence_dimensions=["problem and setting"]),
            ReviewSectionPlan(section_id="taxonomy", title="Method taxonomy", objective="Compare technical approaches.", evidence_dimensions=["method architecture", "training objective"]),
            ReviewSectionPlan(section_id="evidence", title="Experimental evidence", objective="Compare datasets, metrics and reported results.", evidence_dimensions=["experimental evidence"]),
            ReviewSectionPlan(section_id="limitations", title="Limitations and future directions", objective="Synthesize only evidence-backed limitations.", evidence_dimensions=["limitations and cost"]),
        ]


PlanCallable = Callable[[str, list[str], ReviewTemplate, ReviewBudget], LiteratureReviewPlan]
PaperQACallable = Callable[[str, str], PaperQAResult]
SectionWriterCallable = Callable[[ReviewSectionPlan, list[ReviewEvidence]], ReviewSectionDraft]
ReflectionCallable = Callable[[LiteratureReviewPlan, list[ReviewSectionDraft], list[ReviewEvidence]], ReviewReflection]


@dataclass(frozen=True)
class ReviewWriterServices:
    ask_paper: PaperQACallable
    write_section: SectionWriterCallable
    plan: PlanCallable | None = None
    reflect: ReflectionCallable | None = None


class ReviewWriterState(TypedDict, total=False):
    topic: str
    paper_ids: list[str]
    template: ReviewTemplate
    budget: ReviewBudget
    confirmed: bool
    plan: LiteratureReviewPlan
    evidence: list[ReviewEvidence]
    drafts: list[ReviewSectionDraft]
    reflection: ReviewReflection
    reflection_attempts: int
    waiting_confirmation: bool


def build_review_writer_graph(services: ReviewWriterServices):
    """Build the review subgraph with explicit plan confirmation and bounded repair."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("install the agent extra to use literature review workflows") from exc

    def plan(state: ReviewWriterState) -> dict:
        budget = state.get("budget", ReviewBudget())
        value = (
            services.plan(state["topic"], state["paper_ids"], state.get("template", ReviewTemplate.CONFERENCE_STYLE), budget)
            if services.plan
            else DeterministicReviewPlanner().plan(state["topic"], state["paper_ids"], state.get("template", ReviewTemplate.CONFERENCE_STYLE), budget)
        )
        return {"plan": value, "waiting_confirmation": not state.get("confirmed", False)}

    def after_plan(state: ReviewWriterState) -> str:
        return "end" if state.get("waiting_confirmation") else "collect_evidence"

    def collect_evidence(state: ReviewWriterState) -> dict:
        plan_value = state["plan"]
        evidence: list[ReviewEvidence] = []
        remaining_slots = plan_value.budget.max_evidence_tasks
        remaining_sections = len(plan_value.sections)
        for section in plan_value.sections:
            # Allocate the finite evidence budget across sections before choosing
            # papers, so early sections cannot starve later review chapters.
            section_slots = min(
                len(plan_value.paper_ids),
                max(1, remaining_slots // remaining_sections),
            )
            for paper_id in plan_value.paper_ids[:section_slots]:
                question = (
                    f"For the literature review section '{section.title}' ({section.objective}), "
                    f"extract evidence about: {', '.join(section.evidence_dimensions)}. "
                    f"Topic: {plan_value.topic}"
                )
                result = services.ask_paper(question, paper_id)
                evidence.append(
                    ReviewEvidence(
                        section_id=section.section_id,
                        paper_id=paper_id,
                        summary=result.answer,
                        evidence_chunk_ids=result.evidence_chunk_ids,
                        citation_valid=result.citation_valid,
                        abstained=result.abstained,
                    )
                )
                remaining_slots -= 1
            remaining_sections -= 1
            if remaining_slots <= 0:
                return {"evidence": evidence}
        return {"evidence": evidence}

    def write_sections(state: ReviewWriterState) -> dict:
        drafts = [
            services.write_section(section, [item for item in state["evidence"] if item.section_id == section.section_id])
            for section in state["plan"].sections
        ]
        return {"drafts": drafts}

    def reflect(state: ReviewWriterState) -> dict:
        if services.reflect:
            result = services.reflect(state["plan"], state["drafts"], state["evidence"])
        else:
            invalid_sections = sorted({item.section_id for item in state["evidence"] if not item.citation_valid})
            result = ReviewReflection(
                action=ReflectionAction.PASS if not invalid_sections else ReflectionAction.MARK_INSUFFICIENT_EVIDENCE,
                affected_section_ids=invalid_sections,
                reasoning="all collected evidence passed citation validation" if not invalid_sections else "some evidence tasks did not pass citation validation",
            )
        return {"reflection": result, "reflection_attempts": state.get("reflection_attempts", 0) + 1}

    def after_reflection(state: ReviewWriterState) -> str:
        if state["reflection"].action == ReflectionAction.RETRIEVE_MORE and state["reflection_attempts"] < state["plan"].budget.max_reflection_attempts:
            return "collect_evidence"
        return "end"

    graph = StateGraph(ReviewWriterState)
    graph.add_node("plan_review", plan)
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_node("write_sections", write_sections)
    graph.add_node("reflect_review", reflect)
    graph.add_edge(START, "plan_review")
    graph.add_conditional_edges("plan_review", after_plan, {"end": END, "collect_evidence": "collect_evidence"})
    graph.add_edge("collect_evidence", "write_sections")
    graph.add_edge("write_sections", "reflect_review")
    graph.add_conditional_edges("reflect_review", after_reflection, {"collect_evidence": "collect_evidence", "end": END})
    return graph.compile()

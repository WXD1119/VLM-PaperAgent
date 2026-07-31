"""LangGraph subgraph for bounded, evidence-backed paper comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

from paper_agent.planning import PaperQAResult


class ComparisonBudget(BaseModel):
    max_papers: int = Field(default=4, ge=2, le=12)
    max_dimensions: int = Field(default=3, ge=1, le=8)


class ComparisonPlan(BaseModel):
    query: str = Field(min_length=1)
    paper_ids: list[str] = Field(min_length=2)
    dimensions: list[str] = Field(min_length=1)
    budget: ComparisonBudget = Field(default_factory=ComparisonBudget)


class ComparisonCell(BaseModel):
    paper_id: str
    dimension: str
    summary: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_valid: bool
    abstained: bool


class ComparisonMatrix(BaseModel):
    query: str
    paper_ids: list[str]
    dimensions: list[str]
    cells: list[ComparisonCell]


class DeterministicComparisonPlanner:
    """Small, explicit plan that caps downstream PaperQA calls."""

    DEFAULT_DIMENSIONS = ["problem and setting", "method architecture", "experimental evidence"]

    def plan(self, query: str, paper_ids: list[str], budget: ComparisonBudget | None = None) -> ComparisonPlan:
        limits = budget or ComparisonBudget()
        normalized = list(dict.fromkeys(item.strip() for item in paper_ids if item.strip()))
        if len(normalized) < 2:
            raise ValueError("comparison requires at least two distinct paper IDs")
        if len(normalized) > limits.max_papers:
            raise ValueError(f"comparison exceeds max_papers={limits.max_papers}")
        dimensions = self._dimensions(query)[: limits.max_dimensions]
        return ComparisonPlan(query=query, paper_ids=normalized, dimensions=dimensions, budget=limits)

    def _dimensions(self, query: str) -> list[str]:
        text = query.lower()
        candidates: list[str] = []
        if any(token in text for token in ("架构", "architecture", "模块", "连接")):
            candidates.append("method architecture")
        if any(token in text for token in ("训练", "training", "目标", "数据")):
            candidates.append("training objective and data")
        if any(token in text for token in ("实验", "指标", "benchmark", "performance")):
            candidates.append("experimental evidence")
        if any(token in text for token in ("局限", "limitation", "成本", "效率")):
            candidates.append("limitations and cost")
        return list(dict.fromkeys(candidates + self.DEFAULT_DIMENSIONS))


PlanCallable = Callable[[str, list[str], ComparisonBudget], ComparisonPlan]
PaperQACallable = Callable[[str, str], PaperQAResult]


@dataclass(frozen=True)
class ReadingCompareServices:
    ask_paper: PaperQACallable
    plan: PlanCallable | None = None


class ReadingCompareState(TypedDict, total=False):
    query: str
    paper_ids: list[str]
    budget: ComparisonBudget
    plan: ComparisonPlan
    cells: list[ComparisonCell]
    matrix: ComparisonMatrix


def build_reading_compare_graph(services: ReadingCompareServices):
    """Build a bounded Plan-and-Execute comparison graph.

    Individual PaperQA calls remain responsible for retrieval, scope isolation,
    citation validation and refusal. This graph coordinates comparison tasks and
    keeps their evidence IDs in a reusable matrix artifact.
    """

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("install the agent extra to use reading/compare workflows") from exc

    def plan(state: ReadingCompareState) -> dict:
        budget = state.get("budget", ComparisonBudget())
        value = (
            services.plan(state["query"], state["paper_ids"], budget)
            if services.plan
            else DeterministicComparisonPlanner().plan(state["query"], state["paper_ids"], budget)
        )
        return {"plan": value}

    def execute(state: ReadingCompareState) -> dict:
        cells: list[ComparisonCell] = []
        for paper_id in state["plan"].paper_ids:
            for dimension in state["plan"].dimensions:
                task_query = f"For '{dimension}', answer using only this paper: {state['plan'].query}"
                result = services.ask_paper(task_query, paper_id)
                cells.append(
                    ComparisonCell(
                        paper_id=paper_id,
                        dimension=dimension,
                        summary=result.answer,
                        evidence_chunk_ids=result.evidence_chunk_ids,
                        citation_valid=result.citation_valid,
                        abstained=result.abstained,
                    )
                )
        return {"cells": cells}

    def materialize_matrix(state: ReadingCompareState) -> dict:
        plan_value = state["plan"]
        return {"matrix": ComparisonMatrix(query=plan_value.query, paper_ids=plan_value.paper_ids, dimensions=plan_value.dimensions, cells=state["cells"])}

    graph = StateGraph(ReadingCompareState)
    graph.add_node("plan_comparison", plan)
    graph.add_node("execute_paper_qa", execute)
    graph.add_node("materialize_matrix", materialize_matrix)
    graph.add_edge(START, "plan_comparison")
    graph.add_edge("plan_comparison", "execute_paper_qa")
    graph.add_edge("execute_paper_qa", "materialize_matrix")
    graph.add_edge("materialize_matrix", END)
    return graph.compile()

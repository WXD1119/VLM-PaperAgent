"""Deterministic-first router for the three research-workspace agents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .contracts import AgentDispatch, AgentName, CorpusScope


class ResearchRouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    mode: str | None = Field(default=None, max_length=128)
    active_paper_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=20)
    corpus_scope: CorpusScope = CorpusScope.AUTO
    web_expansion_confirmed: bool = False


class ResearchRouteDecision(BaseModel):
    dispatches: list[AgentDispatch] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    needs_clarification: bool = False
    clarification_question: str | None = None


class ResearchRouter:
    """Route explicit modes first and use conservative rules for auto mode."""

    _MODE_MAP = {
        "library": (AgentName.LIBRARY, "manage_papers"),
        "manage_papers": (AgentName.LIBRARY, "manage_papers"),
        "reading": (AgentName.READING_COMPARE, "active_paper_explain"),
        "active_paper_explain": (AgentName.READING_COMPARE, "active_paper_explain"),
        "compare": (AgentName.READING_COMPARE, "multi_paper_compare"),
        "multi_paper_compare": (AgentName.READING_COMPARE, "multi_paper_compare"),
        "related_work": (AgentName.READING_COMPARE, "library_related_work"),
        "library_related_work": (AgentName.READING_COMPARE, "library_related_work"),
        "review": (AgentName.REVIEW_WRITER, "review_write"),
        "review_write": (AgentName.REVIEW_WRITER, "review_write"),
    }

    def route(self, request: ResearchRouteRequest) -> ResearchRouteDecision:
        if request.mode:
            return self._explicit(request)

        text = request.query.lower()
        if any(token in text for token in ("上传", "入库", "解析论文", "管理论文")):
            return self._decision(AgentName.LIBRARY, "manage_papers", CorpusScope.TEMPORARY_WORKSPACE, 0.95)
        if any(token in text for token in ("综述", "survey", "related work", "文献回顾")):
            return self._decision(
                AgentName.REVIEW_WRITER,
                "review_write",
                self._review_scope(request),
                0.92,
                needs_confirmation=not bool(request.selected_paper_ids),
            )
        if any(token in text for token in ("比较", "对比", "区别", "差异", "compare")):
            if len(request.selected_paper_ids) < 2:
                return ResearchRouteDecision(
                    dispatches=[AgentDispatch(agent=AgentName.READING_COMPARE, mode="multi_paper_compare", corpus_scope=CorpusScope.SELECTED_PAPERS)],
                    confidence=0.86,
                    needs_clarification=True,
                    clarification_question="请至少选择两篇论文进行对比。",
                )
            return self._decision(AgentName.READING_COMPARE, "multi_paper_compare", CorpusScope.SELECTED_PAPERS, 0.9)
        related = any(token in text for token in ("有人提过", "是否提过", "相关工作", "论文库", "库里", "已有研究"))
        explain = any(token in text for token in ("解释", "理解", "讲解", "这篇论文", "公式", "图", "表"))
        if related and explain and request.active_paper_id:
            return ResearchRouteDecision(
                dispatches=[
                    AgentDispatch(agent=AgentName.READING_COMPARE, mode="active_paper_explain", corpus_scope=CorpusScope.ACTIVE_PAPER_ONLY),
                    AgentDispatch(agent=AgentName.READING_COMPARE, mode="library_related_work", corpus_scope=CorpusScope.LIBRARY_ONLY, depends_on=["active_paper_explain"]),
                ],
                confidence=0.88,
            )
        if related:
            return self._decision(AgentName.READING_COMPARE, "library_related_work", CorpusScope.LIBRARY_ONLY, 0.82)
        if request.active_paper_id:
            return self._decision(AgentName.READING_COMPARE, "active_paper_explain", CorpusScope.ACTIVE_PAPER_ONLY, 0.72)
        return self._decision(AgentName.READING_COMPARE, "library_related_work", CorpusScope.LIBRARY_ONLY, 0.5)

    def _explicit(self, request: ResearchRouteRequest) -> ResearchRouteDecision:
        item = self._MODE_MAP.get(request.mode or "")
        if item is None:
            return ResearchRouteDecision(
                dispatches=[AgentDispatch(agent=AgentName.READING_COMPARE, mode="library_related_work", corpus_scope=CorpusScope.LIBRARY_ONLY)],
                confidence=0,
                needs_clarification=True,
                clarification_question="未知模式，请选择管理论文、讲解/对比或综述写作。",
            )
        agent, mode = item
        scope = self._scope_for_mode(request, mode)
        needs_confirmation = mode == "review_write" and not bool(request.selected_paper_ids)
        return self._decision(agent, mode, scope, 1.0, needs_confirmation=needs_confirmation)

    @staticmethod
    def _scope_for_mode(request: ResearchRouteRequest, mode: str) -> CorpusScope:
        if mode == "manage_papers":
            return CorpusScope.TEMPORARY_WORKSPACE
        if mode == "active_paper_explain":
            return CorpusScope.ACTIVE_PAPER_ONLY
        if mode == "multi_paper_compare":
            return CorpusScope.SELECTED_PAPERS
        if mode == "review_write":
            return ResearchRouter._review_scope(request)
        return CorpusScope.LIBRARY_ONLY

    @staticmethod
    def _review_scope(request: ResearchRouteRequest) -> CorpusScope:
        if request.web_expansion_confirmed:
            return CorpusScope.WEB_EXPANSION
        return CorpusScope.SELECTED_PAPERS if request.selected_paper_ids else CorpusScope.LIBRARY_ONLY

    @staticmethod
    def _decision(agent: AgentName, mode: str, scope: CorpusScope, confidence: float, *, needs_confirmation: bool = False) -> ResearchRouteDecision:
        return ResearchRouteDecision(
            dispatches=[AgentDispatch(agent=agent, mode=mode, corpus_scope=scope, needs_confirmation=needs_confirmation)],
            confidence=confidence,
        )

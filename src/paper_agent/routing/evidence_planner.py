"""将 QueryPlan 转换为受限、可执行的证据检索计划。"""

from __future__ import annotations

from paper_agent.routing.models import EvidencePlan, EvidenceSubQuestion, QueryIntent, QueryPlan


class EvidencePlanner:
    """不调用 LLM 的初版 Planner；复杂意图显式拆分为最小证据问题。"""

    def plan(self, query: str, route: QueryPlan) -> EvidencePlan:
        sub_questions = [
            EvidenceSubQuestion(
                query=item,
                evidence_types=route.required_evidence_types,
                preferred_sections=route.preferred_sections,
            )
            for item in self._sub_queries(query, route.intent)
        ]
        return EvidencePlan(
            intent=route.intent,
            paper_scope=route.paper_scope,
            sub_questions=sub_questions,
            minimum_evidence_count=self._minimum_evidence(route.intent),
            use_graph=route.use_graph,
            reasons=route.reasons,
        )

    @staticmethod
    def _sub_queries(query: str, intent: QueryIntent) -> list[str]:
        if intent == QueryIntent.MECHANISM:
            return [query, f"{query} architecture", f"{query} training connection"]
        if intent == QueryIntent.COMPARISON:
            return [query, f"{query} method architecture", f"{query} experimental differences"]
        if intent == QueryIntent.MULTI_HOP:
            return [query, f"{query} related papers concepts"]
        return [query]

    @staticmethod
    def _minimum_evidence(intent: QueryIntent) -> int:
        if intent in {QueryIntent.MECHANISM, QueryIntent.COMPARISON, QueryIntent.MULTI_HOP}:
            return 3
        if intent in {QueryIntent.FORMULA_EXPLANATION, QueryIntent.TABLE_LOOKUP, QueryIntent.FIGURE_EXPLANATION}:
            return 2
        return 1

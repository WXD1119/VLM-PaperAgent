"""规则优先的论文问题路由器。"""

from __future__ import annotations

import re

from paper_agent.domain import ChunkKind
from paper_agent.routing.models import QueryIntent, QueryPlan


class QueryRouter:
    """使用可审计规则生成保守的 QueryPlan；后续可接入 LLM 作为低置信度候选。"""

    def route(
        self,
        query: str,
        *,
        paper_id: str | None = None,
        requested_kind: ChunkKind | None = None,
        requested_rerank: bool = True,
    ) -> QueryPlan:
        normalized = query.strip()
        lower = normalized.lower()
        scope = [paper_id] if paper_id else []
        if requested_kind is not None:
            return self._plan(
                QueryIntent.GENERAL,
                scope,
                [requested_kind],
                requested_rerank,
                confidence=1.0,
                reasons=["用户显式限定了证据类型"],
            )
        if self._is_ambiguous_reference(normalized) and not paper_id:
            return QueryPlan(
                intent=QueryIntent.FOLLOW_UP,
                paper_scope=[],
                required_evidence_types=[ChunkKind.TEXT],
                confidence=0.25,
                needs_clarification=True,
                clarification_question="请说明你指的是哪篇论文，以及对应的图、表、公式或方法名称。",
                reasons=["问题包含未绑定到论文上下文的指代"],
            )
        if paper_id and self._is_follow_up_reference(normalized):
            evidence_types = [ChunkKind.TEXT, ChunkKind.FIGURE] if self._matches(lower, "图像", "交互", "architecture", "架构") else [ChunkKind.TEXT]
            return self._plan(
                QueryIntent.FOLLOW_UP,
                scope,
                evidence_types,
                True,
                confidence=0.82,
                reasons=["会话上下文已提供论文范围，问题使用指代或续问表达"],
            )
        if self._matches(lower, "formula", "equation", "式(", "式（", "损失函数", "objective", "插值路径", "conditional probability", "概率", "x_0", "x_1", "插值", "atom positions"):
            return self._plan(QueryIntent.FORMULA_EXPLANATION, scope, [ChunkKind.EQUATION, ChunkKind.TEXT], True, confidence=0.95, sections=["Method"], reasons=["问题要求解释公式或训练目标"])
        if self._matches(lower, "哪些论文", "哪些方法", "关系路径", "出现在哪些", "依赖哪些", "multi-hop"):
            evidence_types = [ChunkKind.TABLE, ChunkKind.TEXT] if self._matches(lower, "benchmark", "指标", "metric") else [ChunkKind.TEXT]
            return self._plan(QueryIntent.MULTI_HOP, scope, evidence_types, True, confidence=0.88, use_graph=True, reasons=["问题要求跨实体或跨论文关系发现"])
        if self._is_table_question(lower):
            return self._plan(QueryIntent.TABLE_LOOKUP, scope, [ChunkKind.TABLE, ChunkKind.TEXT], True, confidence=0.93, sections=["Experiment", "Results"], reasons=["问题要求定位或解释实验表格"])
        if self._matches(lower, "figure", "fig.", "图", "架构图", "框图"):
            return self._plan(QueryIntent.FIGURE_EXPLANATION, scope, [ChunkKind.FIGURE, ChunkKind.TEXT], True, confidence=0.92, sections=["Method"], reasons=["问题要求解释图示及其相邻文字"])
        if self._matches(lower, "区别", "compare", "共同点", "相比", "versus", " vs ", "different", "差异"):
            evidence_types = [ChunkKind.TEXT, ChunkKind.FIGURE] if self._matches(lower, "connect", "连接", "architecture", "架构", "q-former") else [ChunkKind.TEXT]
            return self._plan(QueryIntent.COMPARISON, scope, evidence_types, True, confidence=0.9, use_graph=True, sections=["Method", "Experiment"], reasons=["问题要求跨方法或跨论文比较"])
        if self._matches(lower, "why", "为什么", "动机", "原因", "motivation"):
            return self._plan(QueryIntent.MOTIVATION, scope, [ChunkKind.TEXT], True, confidence=0.86, sections=["Introduction", "Method", "Ablation", "Discussion"], reasons=["问题要求解释设计动机"])
        if self._matches(lower, "表现", "benchmark", "vqa", "scienceqa", "imagenet", "retrieval result", "distribution shift", "结果", "evaluation", "zero-shot", "video-language", "image-text retrieval"):
            return self._plan(QueryIntent.EXPERIMENT, scope, [ChunkKind.TABLE, ChunkKind.TEXT], True, confidence=0.85, sections=["Experiment", "Results"], reasons=["问题要求实验结果或评测设置"])
        if self._matches(lower, "how", "如何", "机制", "connect", "bridge", "architecture", "架构", "交互"):
            evidence_types = [ChunkKind.EQUATION, ChunkKind.TEXT] if "llava" in lower and self._matches(lower, "connect", "连接") else [ChunkKind.TEXT, ChunkKind.FIGURE]
            return self._plan(QueryIntent.MECHANISM, scope, evidence_types, True, confidence=0.82, sections=["Method"], reasons=["问题要求解释方法机制"])
        if self._matches(lower, "来自哪", "哪一页", "where is", "which section"):
            return self._plan(QueryIntent.ATTRIBUTION, scope, [ChunkKind.TEXT], requested_rerank, confidence=0.85, reasons=["问题要求精确来源定位"])
        if self._matches(lower, "what is", "是什么", "含义", "代表什么", "stand for"):
            evidence_types = [ChunkKind.TEXT, ChunkKind.FIGURE] if "perceiver resampler" in lower else [ChunkKind.TEXT]
            return self._plan(QueryIntent.DEFINITION, scope, evidence_types, False, confidence=0.8, reasons=["问题是简短定义性查询，可使用轻量检索"])
        return self._plan(QueryIntent.GENERAL, scope, [ChunkKind.TEXT], requested_rerank, confidence=0.5, reasons=["未命中专门规则，使用保守文本检索策略"])

    @staticmethod
    def _matches(query: str, *terms: str) -> bool:
        return any(term.lower() in query for term in terms)

    @staticmethod
    def _is_ambiguous_reference(query: str) -> bool:
        short = len(re.sub(r"\s+", "", query)) <= 14
        numbered_artifact = bool(re.search(r"^(图\s*\d+|表\s*\d+|式\s*[（(]?\d+)", query))
        has_reference = bool(re.search(r"^(它|这个|那个|继续讲)", query))
        return numbered_artifact or (short and has_reference)

    @staticmethod
    def _is_follow_up_reference(query: str) -> bool:
        return bool(re.search(r"^(它|这些|这个方法|前面|继续讲|它和)", query))

    @staticmethod
    def _is_table_question(query: str) -> bool:
        return bool(
            re.search(r"\btable\b", query)
            or re.search(r"表\s*\d+|表格", query)
            or "结果在哪个表" in query
        )

    @staticmethod
    def _plan(
        intent: QueryIntent,
        scope: list[str],
        evidence_types: list[ChunkKind],
        use_rerank: bool,
        *,
        confidence: float,
        use_graph: bool = False,
        sections: list[str] | None = None,
        reasons: list[str],
    ) -> QueryPlan:
        return QueryPlan(
            intent=intent,
            paper_scope=scope,
            required_evidence_types=evidence_types,
            preferred_sections=sections or [],
            use_rerank=use_rerank,
            use_graph=use_graph,
            use_judge=intent not in {QueryIntent.DEFINITION, QueryIntent.ATTRIBUTION},
            confidence=confidence,
            reasons=reasons,
        )

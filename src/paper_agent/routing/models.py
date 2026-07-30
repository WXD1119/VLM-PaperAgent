"""论文问题路由与证据规划的结构化契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from paper_agent.domain import ChunkKind


class QueryIntent(StrEnum):
    """当前版本支持的论文阅读意图。"""

    DEFINITION = "definition"
    MECHANISM = "mechanism"
    MOTIVATION = "motivation"
    EXPERIMENT = "experiment"
    COMPARISON = "comparison"
    ATTRIBUTION = "attribution"
    FORMULA_EXPLANATION = "formula_explanation"
    TABLE_LOOKUP = "table_lookup"
    FIGURE_EXPLANATION = "figure_explanation"
    MULTI_HOP = "multi_hop"
    FOLLOW_UP = "follow_up"
    GENERAL = "general"


class QueryPlan(BaseModel):
    """路由器的受限决策；仅供系统选择既有能力，不是开放式工具调用计划。"""

    intent: QueryIntent
    paper_scope: list[str] = Field(default_factory=list)
    required_evidence_types: list[ChunkKind] = Field(default_factory=lambda: [ChunkKind.TEXT])
    preferred_sections: list[str] = Field(default_factory=list)
    use_rerank: bool = True
    use_graph: bool = False
    use_judge: bool = True
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: str | None = None
    reasons: list[str] = Field(default_factory=list)


class EvidenceSubQuestion(BaseModel):
    """单个子问题及其最小证据需求。"""

    query: str
    evidence_types: list[ChunkKind]
    preferred_sections: list[str] = Field(default_factory=list)


class EvidencePlan(BaseModel):
    """在检索前明确证据范围，避免 Answer Agent 自由扩张事实。"""

    intent: QueryIntent
    paper_scope: list[str] = Field(default_factory=list)
    sub_questions: list[EvidenceSubQuestion] = Field(min_length=1)
    minimum_evidence_count: int = Field(default=1, ge=1, le=12)
    use_graph: bool = False
    reasons: list[str] = Field(default_factory=list)

"""端到端问答闭环评测：同时衡量证据命中、引用校验、审查与安全拒答。"""

from statistics import mean

from pydantic import BaseModel, Field

from paper_agent.evaluation.golden import RetrievalCase
from paper_agent.evaluation.retrieval import hit_at_k, recall_at_k


class EndToEndResponse(BaseModel):
    """与 API 解耦的端到端评测最小响应结构。"""

    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citation_valid: bool
    abstained: bool
    semantic_gate_enabled: bool = False
    semantic_gate_passed: bool | None = None
    attempts: int = 1
    rewrite_count: int = 0
    refusal_kind: str | None = None
    trace_id: str | None = None


class EndToEndCaseResult(BaseModel):
    query_id: str
    evidence_hit_at_5: float
    evidence_recall_at_5: float
    citation_valid: bool
    abstained: bool
    semantic_gate_passed: bool | None
    attempts: int
    rewrite_count: int
    refusal_kind: str | None = None
    trace_id: str | None = None


class EndToEndEvaluation(BaseModel):
    case_count: int
    evidence_hit_at_5: float
    evidence_recall_at_5: float
    citation_pass_rate: float
    semantic_gate_coverage: float
    semantic_gate_pass_rate: float
    abstention_rate: float
    rewrite_rate: float
    safe_refusal_rate: float
    cases: list[EndToEndCaseResult] = Field(default_factory=list)


def evaluate_end_to_end(
    cases: list[RetrievalCase], responses: dict[str, EndToEndResponse]
) -> EndToEndEvaluation:
    """评测已运行的 API 响应；不把模型答案文本写入评测产物。"""

    if not cases:
        raise ValueError("端到端评测至少需要一个 case")
    results: list[EndToEndCaseResult] = []
    for case in cases:
        response = responses.get(case.query_id)
        if response is None:
            raise ValueError(f"缺少 case 响应：{case.query_id}")
        results.append(
            EndToEndCaseResult(
                query_id=case.query_id,
                evidence_hit_at_5=hit_at_k(response.evidence_chunk_ids, case.relevant_chunk_ids, 5),
                evidence_recall_at_5=recall_at_k(response.evidence_chunk_ids, case.relevant_chunk_ids, 5),
                citation_valid=response.citation_valid,
                abstained=response.abstained,
                semantic_gate_passed=response.semantic_gate_passed,
                attempts=response.attempts,
                rewrite_count=response.rewrite_count,
                refusal_kind=response.refusal_kind,
                trace_id=response.trace_id,
            )
        )
    judged = [case for case in results if case.semantic_gate_passed is not None]
    refusals = [case for case in results if case.abstained]
    return EndToEndEvaluation(
        case_count=len(results),
        evidence_hit_at_5=mean(case.evidence_hit_at_5 for case in results),
        evidence_recall_at_5=mean(case.evidence_recall_at_5 for case in results),
        citation_pass_rate=mean(case.citation_valid for case in results),
        semantic_gate_coverage=len(judged) / len(results),
        semantic_gate_pass_rate=mean(case.semantic_gate_passed for case in judged) if judged else 0.0,
        abstention_rate=mean(case.abstained for case in results),
        rewrite_rate=mean(case.rewrite_count > 0 for case in results),
        safe_refusal_rate=mean(bool(case.refusal_kind) for case in refusals) if refusals else 0.0,
        cases=results,
    )

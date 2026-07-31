from paper_agent.domain import ChunkKind
from paper_agent.routing import EvidencePlan, EvidenceSubQuestion, QueryIntent, QueryRewriter


def test_query_rewriter_normalizes_whitespace_and_deduplicates_without_semantic_expansion():
    plan = EvidencePlan(
        intent=QueryIntent.MECHANISM,
        sub_questions=[
            EvidenceSubQuestion(query="  Q-Former   architecture ", evidence_types=[ChunkKind.TEXT]),
            EvidenceSubQuestion(query="Q-Former architecture", evidence_types=[ChunkKind.TEXT]),
            EvidenceSubQuestion(query="Formula (3)", evidence_types=[ChunkKind.EQUATION]),
        ],
    )
    rewritten = QueryRewriter().rewrite(plan)
    assert [item.query for item in rewritten.sub_questions] == ["Q-Former architecture", "Formula (3)"]

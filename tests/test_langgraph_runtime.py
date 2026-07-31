import pytest

pytest.importorskip("langgraph")

from paper_agent.domain import (
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    ClaimSupportAssessment,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.memory import ContextGuardAction, ContextGuardDecision
from paper_agent.runtime import ResearchGraphRun, ResearchGraphServices, build_research_graph
from paper_agent.orchestration import CorpusScope


def test_langgraph_research_workflow_runs_context_retrieval_and_answer():
    calls: list[str] = []

    def resolve(query: str, paper_id: str | None) -> ContextGuardDecision:
        calls.append("context")
        return ContextGuardDecision(
            action=ContextGuardAction.PROCEED,
            query=query,
            resolved_paper_id=paper_id,
        )

    def retrieve(*_args):
        calls.append("retrieve")
        return [
            type(
                "Hit",
                (),
                {
                    "chunk_id": "chunk-1",
                    "paper_id": "paper-1",
                    "kind": ChunkKind.TEXT,
                    "section_path": ["Method"],
                    "pages": [1],
                    "content": "Evidence",
                },
            )()
        ]

    def evidence(query: str, _hits) -> EvidencePack:
        return EvidencePack(
            query=query,
            items=[
                EvidenceItem(
                    evidence_id="E1",
                    chunk_id="chunk-1",
                    paper_id="paper-1",
                    kind=ChunkKind.TEXT,
                    pages=[1],
                    section_path=["Method"],
                    content="Evidence",
                )
            ],
        )

    def answer(_pack: EvidencePack, _feedback: str | None):
        calls.append("answer")
        result = GroundedAnswer(
            answer="Grounded answer",
            claims=[AnswerClaim(text="Grounded claim", evidence_ids=["E1"])],
        )
        return result, CitationValidation(valid=True, claim_count=1, cited_claim_count=1)

    graph = build_research_graph(
        ResearchGraphServices(
            resolve_context=resolve,
            retrieve=retrieve,
            build_evidence=evidence,
            answer=answer,
            judge=None,
        )
    )
    run = ResearchGraphRun(
        graph.invoke(
            {
                "query": "How does Q-Former work?",
                "paper_id": "paper-1",
                "top_k": 5,
                "use_rerank": True,
                "max_attempts": 2,
            }
        )
    )

    assert calls[0] == "context"
    assert calls.count("retrieve") == 6  # 机制问题拆为 3 个子问题，每个检索文本与图示证据。
    assert calls[-1] == "answer"
    assert run.answer.answer == "Grounded answer"
    assert run.citation_validation.valid is True
    assert run.semantic_gate_enabled is False


def test_langgraph_returns_clarification_without_retrieval():
    graph = build_research_graph(
        ResearchGraphServices(
            resolve_context=lambda query, _paper: ContextGuardDecision(
                action=ContextGuardAction.ASK_CLARIFICATION,
                query=query,
                needs_clarification=True,
                clarification_question="Which paper do you mean?",
            ),
            retrieve=lambda *_args: pytest.fail("retrieval should not run"),
            build_evidence=lambda *_args: pytest.fail("evidence should not run"),
            answer=lambda *_args: pytest.fail("answer should not run"),
        )
    )
    run = ResearchGraphRun(
        graph.invoke(
            {"query": "这个方法呢", "top_k": 5, "use_rerank": True, "max_attempts": 2}
        )
    )
    assert run.answer.abstained is True
    assert run.answer.answer == "Which paper do you mean?"


def test_langgraph_active_paper_scope_filters_library_hits():
    graph = build_research_graph(
        ResearchGraphServices(
            resolve_context=lambda query, paper_id: ContextGuardDecision(
                action=ContextGuardAction.PROCEED, query=query, resolved_paper_id=paper_id
            ),
            retrieve=lambda *_args: [
                type("Hit", (), {"chunk_id": "current", "paper_id": "paper-current", "kind": ChunkKind.TEXT, "section_path": [], "pages": [1], "content": "Current"})(),
                type("Hit", (), {"chunk_id": "library", "paper_id": "paper-library", "kind": ChunkKind.TEXT, "section_path": [], "pages": [1], "content": "Library"})(),
            ],
            build_evidence=lambda query, hits: EvidencePack(
                query=query,
                items=[
                    EvidenceItem(
                        evidence_id=f"E{index}",
                        chunk_id=hit.chunk_id,
                        paper_id=hit.paper_id,
                        kind=hit.kind,
                        pages=hit.pages,
                        content=hit.content,
                    )
                    for index, hit in enumerate(hits, start=1)
                ],
            ),
            answer=lambda pack, _feedback: (
                GroundedAnswer(answer="Grounded", claims=[AnswerClaim(text="Grounded", evidence_ids=["E1"])]),
                CitationValidation(valid=True, claim_count=1, cited_claim_count=1),
            ),
        )
    )
    run = ResearchGraphRun(
        graph.invoke(
            {
                "query": "Explain this paper",
                "paper_id": "paper-current",
                "corpus_scope": CorpusScope.ACTIVE_PAPER_ONLY,
                "top_k": 2,
                "use_rerank": True,
                "max_attempts": 1,
            }
        )
    )
    assert [item.paper_id for item in run.evidence_pack.items] == ["paper-current"]
    assert run.scope_resolution.scope == CorpusScope.ACTIVE_PAPER_ONLY


def test_langgraph_rewrites_once_then_refuses_when_judge_keeps_rejecting():
    calls: list[str] = []
    pack = EvidencePack(
        query="Question",
        items=[EvidenceItem(evidence_id="E1", chunk_id="c1", paper_id="p1", kind=ChunkKind.TEXT, pages=[1], content="Evidence")],
    )

    def answer(_pack, feedback):
        calls.append("rewrite" if feedback else "initial")
        return (
            GroundedAnswer(answer="Claim", claims=[AnswerClaim(text="Claim", evidence_ids=["E1"])]),
            CitationValidation(valid=True, claim_count=1, cited_claim_count=1),
        )

    report = SemanticCitationReport(
        assessments=[ClaimSupportAssessment(claim_index=1, verdict=SupportVerdict.UNSUPPORTED, reasoning_summary="not entailed")]
    )
    graph = build_research_graph(
        ResearchGraphServices(
            resolve_context=lambda query, paper_id: ContextGuardDecision(action=ContextGuardAction.PROCEED, query=query, resolved_paper_id=paper_id),
            retrieve=lambda *_args: [
                type(
                    "Hit",
                    (),
                    {
                        "chunk_id": "c1",
                        "paper_id": "p1",
                        "kind": ChunkKind.TEXT,
                        "section_path": [],
                        "pages": [1],
                        "content": "Evidence",
                    },
                )()
            ],
            build_evidence=lambda *_args: pack,
            answer=answer,
            judge=lambda *_args: report,
        )
    )
    run = ResearchGraphRun(graph.invoke({"query": "Question", "top_k": 1, "max_attempts": 2}))
    assert calls == ["initial", "rewrite"]
    assert run.answer.abstained is True
    assert run.refusal_kind == "semantic_support_failed"
    assert run.rewrite_count == 1

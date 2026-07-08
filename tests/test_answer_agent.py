import pytest

from paper_agent.agents.answer import (
    AnswerAgent,
    CitationValidator,
    SemanticCitationJudge,
    build_evidence_pack,
)
from paper_agent.domain import (
    AnswerClaim,
    ChunkKind,
    ClaimSupportAssessment,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.retrieval.reranker import RerankedHit
from paper_agent.llm.client import TransformersStructuredClient


def hit() -> RerankedHit:
    return RerankedHit(
        chunk_id="chunk-1", paper_id="paper-1", kind=ChunkKind.TEXT, score=0.9,
        retrieval_score=0.2, retrieval_rank=1, pages=[2], section_path=["Method"],
        content="Q-Former uses learnable queries to extract visual features.",
    )


class FakeClient:
    def generate_structured(self, prompt, response_model):
        assert "[E1]" in prompt
        return response_model(
            answer="Q-Former uses learnable queries [E1].",
            claims=[AnswerClaim(text="It uses learnable queries.", evidence_ids=["E1"])],
        )


def test_answer_agent_builds_traceable_answer():
    pack = build_evidence_pack("How does Q-Former work?", [hit()])
    answer, report = AnswerAgent(FakeClient()).answer(pack)
    assert answer.claims[0].evidence_ids == ["E1"]
    assert report.valid


def test_validator_rejects_unknown_evidence():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(
        answer="unsupported",
        claims=[AnswerClaim(text="claim", evidence_ids=["E99"])],
    )
    report = CitationValidator().validate(answer, pack)
    assert not report.valid
    assert "E99" in report.errors[0]


def test_abstention_requires_reason():
    with pytest.raises(ValueError):
        GroundedAnswer(answer="", abstained=True)


def test_local_client_extracts_fenced_json():
    content = 'preface\n```json\n{"answer": "ok"}\n```'
    assert TransformersStructuredClient._extract_json(content) == '{"answer": "ok"}'


class FakeJudgeClient:
    def generate_structured(self, prompt, response_model):
        assert "Claim 1" in prompt and "[E1]" in prompt
        return SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.SUPPORTED,
                    evidence_ids=["E1"],
                    reasoning_summary="The evidence states the claim directly.",
                )
            ]
        )


def test_semantic_judge_assesses_each_claim():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(
        answer="answer",
        claims=[AnswerClaim(text="It uses learnable queries.", evidence_ids=["E1"])],
    )
    report = SemanticCitationJudge(FakeJudgeClient()).evaluate(answer, pack)
    assert report.all_supported


def test_semantic_judge_skips_abstention():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(answer="unknown", abstained=True, abstention_reason="No evidence")
    assert SemanticCitationJudge(FakeJudgeClient()).evaluate(answer, pack).assessments == []

import pytest

from paper_agent.agents.answer import (
    AnswerAgent,
    CitationValidator,
    SemanticCitationJudge,
    build_evidence_pack,
)
from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    ClaimSupportAssessment,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.retrieval.reranker import RerankedHit
from paper_agent.llm.client import GlmStructuredClient, TransformersStructuredClient


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


def test_local_client_extracts_first_balanced_json():
    content = '<answer>{"answer": "ok", "text": "{not a brace}"}</answer>\nextra {"ignored": true}'
    assert TransformersStructuredClient._extract_json(content) == '{"answer": "ok", "text": "{not a brace}"}'


def test_glm_client_extracts_json_before_extra_data():
    content = '<answer>{"assessments": []}</answer>\n<think>extra explanation</think>\n{"ignored": true}'
    assert GlmStructuredClient._extract_json(content) == {"assessments": []}


def test_glm_client_rejects_copied_json_schema():
    content = '<answer>{"$defs": {}, "properties": {}, "title": "ClaimSupportAssessment", "type": "object"}</answer>'
    with pytest.raises(ValueError, match="JSON Schema"):
        GlmStructuredClient._extract_json(content)


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


class IncompleteThenSingleJudgeClient:
    def __init__(self):
        self.calls = 0

    def generate_structured(self, prompt, response_model):
        self.calls += 1
        if self.calls == 1:
            return SemanticCitationReport(
                assessments=[
                    ClaimSupportAssessment(
                        claim_index=1,
                        verdict=SupportVerdict.SUPPORTED,
                        evidence_ids=["E1"],
                        reasoning_summary="Only the first claim was judged.",
                    )
                ]
            )
        claim_index = 1 if "claim_index=1" in prompt else 2
        return ClaimSupportAssessment(
            claim_index=claim_index,
            verdict=SupportVerdict.SUPPORTED,
            evidence_ids=["E1"],
            reasoning_summary="The single-claim fallback judged this claim.",
        )


def test_semantic_judge_assesses_each_claim():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(
        answer="answer",
        claims=[AnswerClaim(text="It uses learnable queries.", evidence_ids=["E1"])],
    )
    report = SemanticCitationJudge(FakeJudgeClient()).evaluate(answer, pack)
    assert report.all_supported


def test_semantic_judge_falls_back_when_batch_is_incomplete():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(
        answer="answer",
        claims=[
            AnswerClaim(text="It uses learnable queries.", evidence_ids=["E1"]),
            AnswerClaim(text="It extracts visual features.", evidence_ids=["E1"]),
        ],
    )
    client = IncompleteThenSingleJudgeClient()
    report = SemanticCitationJudge(client).evaluate(answer, pack)
    assert [item.claim_index for item in report.assessments] == [1, 2]
    assert report.all_supported
    assert client.calls == 3


def test_semantic_judge_skips_abstention():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(answer="unknown", abstained=True, abstention_reason="No evidence")
    assert SemanticCitationJudge(FakeJudgeClient()).evaluate(answer, pack).assessments == []


def test_answer_bundle_round_trip():
    pack = build_evidence_pack("question", [hit()])
    answer = GroundedAnswer(
        answer="answer",
        claims=[AnswerClaim(text="It uses learnable queries.", evidence_ids=["E1"])],
    )
    validation = CitationValidator().validate(answer, pack)
    bundle = AnswerBundle(
        evidence_pack=pack,
        answer=answer,
        citation_validation=validation,
        generator_model="local-model",
    )
    restored = AnswerBundle.model_validate_json(bundle.model_dump_json())
    assert restored.answer.claims[0].evidence_ids == ["E1"]

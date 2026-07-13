import importlib.util
from pathlib import Path

import pytest
import pydantic

if not hasattr(pydantic, "model_validator"):
    pytest.skip("ask CLI tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    ClaimSupportAssessment,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)


def load_ask_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ask.py"
    spec = importlib.util.spec_from_file_location("ask_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evidence_pack() -> EvidencePack:
    return EvidencePack(
        query="How does Q-Former work?",
        items=[
            EvidenceItem(
                evidence_id="E1",
                chunk_id="chunk-1",
                paper_id="paper-1",
                kind=ChunkKind.TEXT,
                pages=[2],
                section_path=["Method"],
                content="Q-Former uses learnable queries to extract visual features.",
            )
        ],
    )


def test_render_bundle_shows_answer_claims_and_evidence():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=evidence_pack(),
        answer=GroundedAnswer(
            answer="Q-Former uses learnable queries.",
            claims=[
                AnswerClaim(
                    text="Q-Former uses learnable queries.",
                    evidence_ids=["E1"],
                )
            ],
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=1,
            cited_claim_count=1,
        ),
        generator_model="local-model",
    )

    rendered = ask.render_bundle(bundle, content_chars=40)

    assert "# Answer" in rendered
    assert "Q-Former uses learnable queries." in rendered
    assert "1. Q-Former uses learnable queries. [E1]" in rendered
    assert "Status: PASS" in rendered
    assert "[E1] paper-1 | text | pages=[2]" in rendered


def test_render_memory_policy_hint_recommends_user_confirmation_for_supported_answer():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=evidence_pack(),
        answer=GroundedAnswer(
            answer="Q-Former uses learnable queries.",
            claims=[
                AnswerClaim(
                    text="Q-Former uses learnable queries.",
                    evidence_ids=["E1"],
                )
            ],
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=1,
            cited_claim_count=1,
        ),
        generator_model="local-model",
    )
    report = SemanticCitationReport(
        assessments=[
            ClaimSupportAssessment(
                claim_index=1,
                verdict=SupportVerdict.SUPPORTED,
                evidence_ids=["E1"],
                reasoning_summary="supported",
            )
        ]
    )

    rendered = ask.render_memory_policy_hint(bundle, report)

    assert "# Memory policy" in rendered
    assert "artifact_status: kept" in rendered
    assert "paper_kg_status: not_written" in rendered
    assert "promotion_verdict: ask_user" in rendered


def test_render_memory_summary_hides_policy_details_for_supported_answer():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=evidence_pack(),
        answer=GroundedAnswer(
            answer="Q-Former uses learnable queries.",
            claims=[
                AnswerClaim(
                    text="Q-Former uses learnable queries.",
                    evidence_ids=["E1"],
                )
            ],
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=1,
            cited_claim_count=1,
        ),
        generator_model="local-model",
    )
    report = SemanticCitationReport(
        assessments=[
            ClaimSupportAssessment(
                claim_index=1,
                verdict=SupportVerdict.SUPPORTED,
                evidence_ids=["E1"],
                reasoning_summary="supported",
            )
        ]
    )

    rendered = ask.render_memory_summary(bundle, report)

    assert "# Memory" in rendered
    assert "answer saved as artifact" in rendered
    assert "not written to Paper KG" in rendered
    assert "ask user before long-term archiving" in rendered
    assert "promotion_verdict" not in rendered
    assert "artifact_status" not in rendered


def test_render_memory_policy_hint_rejects_abstained_answer():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=evidence_pack(),
        answer=GroundedAnswer(
            answer="The evidence is insufficient.",
            claims=[],
            abstained=True,
            abstention_reason="insufficient evidence",
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=0,
            cited_claim_count=0,
        ),
        generator_model="local-model",
    )

    rendered = ask.render_memory_policy_hint(bundle)

    assert "promotion_verdict: reject" in rendered
    assert "answer abstained" in rendered
    assert "keep as artifact only" in rendered


def test_render_memory_summary_keeps_abstained_answer_as_artifact_only():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=evidence_pack(),
        answer=GroundedAnswer(
            answer="The evidence is insufficient.",
            claims=[],
            abstained=True,
            abstention_reason="insufficient evidence",
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=0,
            cited_claim_count=0,
        ),
        generator_model="local-model",
    )

    rendered = ask.render_memory_summary(bundle)

    assert "# Memory" in rendered
    assert "keep as artifact only; do not archive" in rendered
    assert "promotion_verdict" not in rendered


class FeedbackAwareAgent:
    def __init__(self):
        self.calls = 0

    def answer(self, pack, feedback=None):
        self.calls += 1
        claim_text = "unsupported broad claim" if feedback is None else "supported revised claim"
        answer = GroundedAnswer(
            answer=claim_text,
            claims=[AnswerClaim(text=claim_text, evidence_ids=["E1"])],
        )
        validation = CitationValidation(valid=True, claim_count=1, cited_claim_count=1)
        return answer, validation


class FirstFailThenPassJudge:
    def __init__(self):
        self.calls = 0

    def evaluate(self, answer, pack):
        self.calls += 1
        verdict = (
            SupportVerdict.UNSUPPORTED
            if self.calls == 1
            else SupportVerdict.SUPPORTED
        )
        return SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=verdict,
                    evidence_ids=["E1"] if verdict == SupportVerdict.SUPPORTED else [],
                    reasoning_summary="judge feedback",
                )
            ]
        )


class AlwaysFailJudge:
    def evaluate(self, answer, pack):
        return SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.UNSUPPORTED,
                    evidence_ids=[],
                    reasoning_summary="not supported",
                )
            ]
        )


def test_semantic_gate_retries_until_supported():
    ask = load_ask_module()
    agent = FeedbackAwareAgent()
    judge = FirstFailThenPassJudge()

    bundle, report, attempts, gate_enabled = ask.answer_with_optional_semantic_gate(
        agent=agent,
        pack=evidence_pack(),
        judge=judge,
        max_attempts=2,
    )

    assert gate_enabled
    assert attempts == 2
    assert agent.calls == 2
    assert report.all_supported
    assert bundle.answer.answer == "supported revised claim"


def test_semantic_gate_abstains_after_failed_attempts():
    ask = load_ask_module()

    bundle, report, attempts, gate_enabled = ask.answer_with_optional_semantic_gate(
        agent=FeedbackAwareAgent(),
        pack=evidence_pack(),
        judge=AlwaysFailJudge(),
        max_attempts=1,
    )

    assert gate_enabled
    assert attempts == 1
    assert not report.all_supported
    assert bundle.answer.abstained
    assert bundle.citation_validation.valid

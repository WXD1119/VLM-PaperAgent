import importlib.util
from pathlib import Path

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

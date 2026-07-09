import importlib.util
from pathlib import Path

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
)


def load_ask_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ask.py"
    spec = importlib.util.spec_from_file_location("ask_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_bundle_shows_answer_claims_and_evidence():
    ask = load_ask_module()
    bundle = AnswerBundle(
        evidence_pack=EvidencePack(
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
        ),
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

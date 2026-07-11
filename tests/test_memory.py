import pytest
import pydantic

if not hasattr(pydantic, "model_validator"):
    pytest.skip("memory tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    ClaimSupportAssessment,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.memory import (
    Episode,
    EpisodicMemoryStore,
    PromotionPolicy,
    PromotionVerdict,
    UserProfileStore,
    list_answer_artifacts,
)


def sample_answer(abstained: bool = False) -> AnswerBundle:
    claims = [] if abstained else [AnswerClaim(text="LLaVA uses a projection layer.", evidence_ids=["E1"])]
    return AnswerBundle(
        evidence_pack=EvidencePack(
            query="How does LLaVA connect vision and language?",
            items=[
                EvidenceItem(
                    evidence_id="E1",
                    chunk_id="chunk-1",
                    paper_id="paper-llava",
                    kind=ChunkKind.TEXT,
                    pages=[4],
                    section_path=["Visual Instruction Tuning", "4.1 Architecture"],
                    content="A projection matrix maps visual features into language embeddings.",
                )
            ],
        ),
        answer=GroundedAnswer(
            answer="LLaVA connects vision and language with a projection layer."
            if not abstained
            else "The evidence is insufficient.",
            claims=claims,
            abstained=abstained,
            abstention_reason="insufficient evidence" if abstained else None,
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=len(claims),
            cited_claim_count=len(claims),
        ),
        generator_model="test-model",
    )


def test_list_answer_artifacts_summarizes_saved_answers(tmp_path):
    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    (answers_dir / "llava.answer.json").write_text(
        sample_answer().model_dump_json(indent=2),
        encoding="utf-8",
    )

    summaries = list_answer_artifacts(answers_dir)

    assert len(summaries) == 1
    assert summaries[0].query == "How does LLaVA connect vision and language?"
    assert summaries[0].claim_count == 1
    assert summaries[0].evidence_count == 1
    assert summaries[0].generator_model == "test-model"


def test_episodic_memory_appends_events_without_touching_graph(tmp_path):
    store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")

    episode = store.append(
        Episode(
            event_type="paper_added",
            summary="Added LLaVA.pdf to wxd workspace",
            payload={"paper_id": "paper-llava"},
        )
    )

    loaded = store.list()
    assert loaded == [episode]
    assert loaded[0].payload["paper_id"] == "paper-llava"


def test_user_profile_store_keeps_stable_preferences_outside_graph(tmp_path):
    store = UserProfileStore(tmp_path / "profile.json")

    profile = store.set("default_workspace_id", "ws_wxd_demo")
    profile = store.set("preferred_gpu", "cuda:2")

    assert profile.default_workspace_id == "ws_wxd_demo"
    assert profile.preferences["preferred_gpu"] == "cuda:2"
    assert store.load().preferences["preferred_gpu"] == "cuda:2"


def test_promotion_policy_rejects_abstentions_and_asks_user_for_supported_answers():
    supported_report = SemanticCitationReport(
        assessments=[
            ClaimSupportAssessment(
                claim_index=1,
                verdict=SupportVerdict.SUPPORTED,
                evidence_ids=["E1"],
                reasoning_summary="directly supported",
            )
        ]
    )
    policy = PromotionPolicy()

    supported = policy.decide(sample_answer(), supported_report)
    rejected = policy.decide(sample_answer(abstained=True), supported_report)

    assert supported.verdict == PromotionVerdict.ASK_USER
    assert rejected.verdict == PromotionVerdict.REJECT
    assert "answer abstained" in rejected.reasons

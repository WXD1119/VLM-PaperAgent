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
    ContextGuard,
    ContextGuardAction,
    Episode,
    EpisodicMemoryStore,
    MemoryAction,
    MemoryCandidate,
    MemoryCandidateKind,
    MemoryPolicy,
    MemoryTarget,
    PromotionPolicy,
    PromotionVerdict,
    SessionMemoryStore,
    SessionState,
    UserProfileStore,
    build_memory_summary,
    list_answer_artifacts,
)
from scripts.memory_apply import apply_memory_decision


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


def test_session_profile_and_episode_memory_can_be_read_together(tmp_path):
    session_store = SessionMemoryStore(tmp_path / "session.json")
    profile_store = UserProfileStore(tmp_path / "profile.json")
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")

    session_store.update(
        current_paper_id="paper-llava",
        last_answer_path="artifacts/answers/llava.answer.json",
    )
    profile_store.set("default_workspace_id", "ws_wxd_demo")
    episode_store.log(
        "paper_added_to_workspace",
        "Added LLaVA paper",
        {"paper_id": "paper-llava"},
    )

    assert session_store.load().current_paper_id == "paper-llava"
    assert profile_store.load().default_workspace_id == "ws_wxd_demo"
    assert episode_store.list(limit=1)[0].event_type == "paper_added_to_workspace"


def test_memory_show_payload_shape_can_be_assembled(tmp_path):
    session_store = SessionMemoryStore(tmp_path / "session.json")
    profile_store = UserProfileStore(tmp_path / "profile.json")
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")
    session = session_store.update(last_query="What is LLaVA?")
    profile = profile_store.set("preferred_language", "zh")
    episode_store.log("answer_generated", "Generated LLaVA answer")
    episodes = episode_store.list(limit=5)

    payload = {
        "session": session.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "recent_episodes": [episode.model_dump(mode="json") for episode in episodes],
    }

    assert payload["session"]["last_query"] == "What is LLaVA?"
    assert payload["profile"]["preferred_language"] == "zh"
    assert payload["recent_episodes"][0]["event_type"] == "answer_generated"


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


def test_build_memory_summary_is_safe_for_ui_and_api():
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

    supported = build_memory_summary(sample_answer(), supported_report)
    abstained = build_memory_summary(sample_answer(abstained=True), supported_report)

    assert supported.status == "answer saved as artifact; not written to Paper KG"
    assert supported.paper_kg_written is False
    assert supported.recommendation == "ask user before long-term archiving"
    assert supported.requires_user_confirmation is True
    assert abstained.recommendation == "keep as artifact only; do not archive"
    assert abstained.requires_user_confirmation is False


def test_context_guard_constrains_ambiguous_followup_to_session_paper():
    state = SessionState(current_paper_id="paper_llava")

    decision = ContextGuard().decide("这个方法怎么连接视觉和语言？", session=state)

    assert decision.action == ContextGuardAction.CONSTRAIN
    assert decision.resolved_paper_id == "paper_llava"
    assert not decision.needs_clarification
    assert decision.warnings


def test_context_guard_asks_when_ambiguous_followup_has_no_context():
    decision = ContextGuard().decide("这个方法怎么样？")

    assert decision.action == ContextGuardAction.ASK_CLARIFICATION
    assert decision.needs_clarification
    assert "Which paper" in decision.clarification_question


def test_context_guard_respects_explicit_paper_id_over_session_memory():
    state = SessionState(current_paper_id="paper_blip2")

    decision = ContextGuard().decide(
        "这个方法怎么样？",
        explicit_paper_id="paper_llava",
        session=state,
    )

    assert decision.action == ContextGuardAction.PROCEED
    assert decision.resolved_paper_id == "paper_llava"
    assert decision.context_notes == ["explicit paper_id provided by user"]


def test_context_guard_does_not_constrain_clear_new_question_to_old_session_paper():
    state = SessionState(current_paper_id="paper_llava")

    decision = ContextGuard().decide("What training objective does CLIP use?", session=state)

    assert decision.action == ContextGuardAction.PROCEED
    assert decision.resolved_paper_id is None
    assert not decision.needs_clarification


def test_memory_policy_routes_candidates_to_separate_memory_layers():
    policy = MemoryPolicy()

    assert policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.TASK_STATE, content="current paper is LLaVA")
    ).target == MemoryTarget.SESSION
    assert policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.PROJECT_EVENT, content="added LLaVA.pdf")
    ).target == MemoryTarget.EPISODIC
    assert policy.decide(
        MemoryCandidate(
            kind=MemoryCandidateKind.USER_PREFERENCE,
            content="prefer Chinese explanations",
            explicit_user_request=True,
        )
    ).target == MemoryTarget.PROFILE
    assert policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.ANSWER_ARTIFACT, content="llava.answer.json")
    ).target == MemoryTarget.ARTIFACT


def test_memory_policy_keeps_paper_content_out_of_agent_memory_and_ignores_chat():
    policy = MemoryPolicy()

    paper_content = policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.PAPER_CONTENT, content="LLaVA architecture")
    )
    casual_chat = policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.CASUAL_CHAT, content="thanks")
    )
    inferred_preference = policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.USER_PREFERENCE, content="maybe likes short answers")
    )

    assert paper_content.action == MemoryAction.ROUTE_OUTSIDE_MEMORY
    assert paper_content.target == MemoryTarget.PAPER_KG
    assert casual_chat.action == MemoryAction.IGNORE
    assert casual_chat.target == MemoryTarget.NONE
    assert inferred_preference.action == MemoryAction.ASK_USER
    assert inferred_preference.requires_user_confirmation


def test_memory_policy_approved_targets_can_be_written_by_stores(tmp_path):
    policy = MemoryPolicy()
    session_store = SessionMemoryStore(tmp_path / "session.json")
    profile_store = UserProfileStore(tmp_path / "profile.json")
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")

    task = MemoryCandidate(kind=MemoryCandidateKind.TASK_STATE, content="reading LLaVA")
    event = MemoryCandidate(kind=MemoryCandidateKind.PROJECT_EVENT, content="added LLaVA")
    preference = MemoryCandidate(
        kind=MemoryCandidateKind.USER_PREFERENCE,
        content="中文解释",
        explicit_user_request=True,
    )

    assert policy.decide(task).action == MemoryAction.WRITE
    session_store.update(active_task=task.content)
    assert policy.decide(event).action == MemoryAction.WRITE
    episode_store.log("memory_event", event.content)
    assert policy.decide(preference).action == MemoryAction.WRITE
    profile_store.set("preferred_language", preference.content)

    assert session_store.load().active_task == "reading LLaVA"
    assert episode_store.list(limit=1)[0].summary == "added LLaVA"
    assert profile_store.load().preferred_language == "中文解释"


def test_memory_policy_blocked_targets_are_not_written_by_default(tmp_path):
    policy = MemoryPolicy()
    episode_store = EpisodicMemoryStore(tmp_path / "episodes.jsonl")
    profile_store = UserProfileStore(tmp_path / "profile.json")

    paper_content = policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.PAPER_CONTENT, content="LLaVA architecture")
    )
    inferred_preference = policy.decide(
        MemoryCandidate(kind=MemoryCandidateKind.USER_PREFERENCE, content="likes concise answers")
    )

    assert paper_content.action == MemoryAction.ROUTE_OUTSIDE_MEMORY
    assert inferred_preference.action == MemoryAction.ASK_USER
    assert episode_store.list() == []
    assert profile_store.load().preferences == {}


def test_apply_memory_decision_writes_only_policy_approved_records(tmp_path):
    policy = MemoryPolicy()
    session_path = tmp_path / "session.json"
    profile_path = tmp_path / "profile.json"
    episodes_path = tmp_path / "episodes.jsonl"
    event = MemoryCandidate(
        kind=MemoryCandidateKind.PROJECT_EVENT,
        content="added LLaVA",
        metadata={"event_type": "paper_added"},
    )
    chat = MemoryCandidate(kind=MemoryCandidateKind.CASUAL_CHAT, content="thanks")

    event_result = apply_memory_decision(
        event,
        policy.decide(event),
        session_path=str(session_path),
        profile_path=str(profile_path),
        episodes_path=str(episodes_path),
    )
    chat_result = apply_memory_decision(
        chat,
        policy.decide(chat),
        session_path=str(session_path),
        profile_path=str(profile_path),
        episodes_path=str(episodes_path),
    )

    assert event_result["written"] is True
    assert EpisodicMemoryStore(episodes_path).list(limit=1)[0].event_type == "paper_added"
    assert chat_result["written"] is False
    assert SessionMemoryStore(session_path).load().active_task is None


def test_apply_memory_decision_writes_session_context_fields(tmp_path):
    session_path = tmp_path / "session.json"
    profile_path = tmp_path / "profile.json"
    episodes_path = tmp_path / "episodes.jsonl"
    candidate = MemoryCandidate(
        kind=MemoryCandidateKind.TASK_STATE,
        content="reading LLaVA",
        metadata={
            "current_paper_id": "paper_llava",
            "current_workspace_id": "ws_demo",
            "last_query": "What is LLaVA?",
        },
    )
    decision = MemoryPolicy().decide(candidate)

    result = apply_memory_decision(
        candidate,
        decision,
        session_path=str(session_path),
        profile_path=str(profile_path),
        episodes_path=str(episodes_path),
    )

    session = SessionMemoryStore(session_path).load()
    assert result["written"] is True
    assert session.current_paper_id == "paper_llava"
    assert session.current_workspace_id == "ws_demo"
    assert session.last_query == "What is LLaVA?"

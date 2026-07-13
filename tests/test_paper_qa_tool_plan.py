from dataclasses import dataclass

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("paper QA tool plan tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import ChunkKind
from paper_agent.memory import (
    ConversationTurn,
    HeuristicConversationSummarizer,
    SessionMemoryStore,
    SummaryMemoryStore,
    UserProfileStore,
)
from paper_agent.tools import ToolStatus
from paper_agent.tools.paper_qa import PaperQAToolPlan


@dataclass
class FakeHit:
    chunk_id: str
    paper_id: str
    kind: ChunkKind = ChunkKind.TEXT


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=5, paper_id=None, kind=None):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "paper_id": paper_id,
                "kind": kind,
            }
        )
        return [FakeHit(chunk_id="c1", paper_id=paper_id or "paper_any")]


def test_paper_qa_tool_plan_constrains_ambiguous_query_to_session_paper(tmp_path):
    session = SessionMemoryStore(tmp_path / "session.json")
    profile = UserProfileStore(tmp_path / "profile.json")
    session.update(current_paper_id="paper_llava")
    retriever = FakeRetriever()
    plan = PaperQAToolPlan(
        retriever=retriever,
        session_store=session,
        profile_store=profile,
    )

    result = plan.run("这个方法怎么连接视觉和语言？", top_k=3)

    assert result.context.action.value == "constrain"
    assert result.context.resolved_paper_id == "paper_llava"
    assert retriever.calls[0]["paper_id"] == "paper_llava"
    assert retriever.calls[0]["top_k"] == 3
    assert result.retrieved_count == 1
    assert result.session_written
    assert session.load().last_query == "这个方法怎么连接视觉和语言？"
    assert {item.call_id: item.status for item in result.tool_results}["retrieve"] == ToolStatus.SUCCESS


def test_paper_qa_tool_plan_asks_clarification_without_calling_tools(tmp_path):
    session = SessionMemoryStore(tmp_path / "session.json")
    profile = UserProfileStore(tmp_path / "profile.json")
    retriever = FakeRetriever()
    plan = PaperQAToolPlan(
        retriever=retriever,
        session_store=session,
        profile_store=profile,
    )

    result = plan.run("这个方法怎么样？")

    assert result.needs_clarification
    assert result.context.needs_clarification
    assert result.tool_results == []
    assert retriever.calls == []
    assert session.load().last_query is None


def test_paper_qa_tool_plan_does_not_bind_clear_new_question_to_old_session(tmp_path):
    session = SessionMemoryStore(tmp_path / "session.json")
    profile = UserProfileStore(tmp_path / "profile.json")
    session.update(current_paper_id="paper_llava")
    retriever = FakeRetriever()
    plan = PaperQAToolPlan(
        retriever=retriever,
        session_store=session,
        profile_store=profile,
    )

    result = plan.run("What training objective does CLIP use?")

    assert result.context.action.value == "proceed"
    assert result.context.resolved_paper_id is None
    assert retriever.calls[0]["paper_id"] is None


def test_paper_qa_tool_plan_recalls_summary_memory(tmp_path):
    session = SessionMemoryStore(tmp_path / "session.json")
    profile = UserProfileStore(tmp_path / "profile.json")
    summaries = SummaryMemoryStore(tmp_path / "summaries.jsonl")
    summary = HeuristicConversationSummarizer().summarize(
        [
            ConversationTurn(
                role="event",
                content="Decided LLaVA memory recall should use summary memory.",
                metadata={"paper_id": "paper_llava"},
            )
        ]
    )
    summaries.append(summary)
    retriever = FakeRetriever()
    plan = PaperQAToolPlan(
        retriever=retriever,
        session_store=session,
        profile_store=profile,
        summary_store=summaries,
    )

    result = plan.run("How should LLaVA memory recall work?")

    assert result.memory_recall_count == 1
    recall = {item.call_id: item for item in result.tool_results}["memory_recall"]
    assert recall.status == ToolStatus.SUCCESS
    assert recall.output[0]["summary_id"] == summary.summary_id

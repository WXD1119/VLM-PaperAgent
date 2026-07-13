from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from paper_agent.domain import ChunkKind
from paper_agent.memory import (
    ConversationTurn,
    HeuristicConversationSummarizer,
    SessionMemoryStore,
    SummaryMemoryStore,
    UserProfileStore,
)
from paper_agent.tools.paper_qa import PaperQAToolPlan


@dataclass
class DemoHit:
    chunk_id: str
    paper_id: str
    kind: ChunkKind = ChunkKind.TEXT


class DemoRetriever:
    def search(self, query, top_k=5, paper_id=None, kind=None):
        return [
            DemoHit(
                chunk_id="demo_chunk_1",
                paper_id=paper_id or "unconstrained",
            )
        ]


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        session = SessionMemoryStore(root / "session.json")
        profile = UserProfileStore(root / "profile.json")
        summaries = SummaryMemoryStore(root / "summaries.jsonl")

        session.update(current_paper_id="paper_dc8bace378a282ea")
        profile.set("preferred_language", "zh")
        summary = HeuristicConversationSummarizer().summarize(
            [
                ConversationTurn(
                    role="event",
                    content="Decided LLaVA memory recall should use summary memory.",
                    metadata={"paper_id": "paper_dc8bace378a282ea"},
                )
            ]
        )
        summaries.append(summary)

        result = PaperQAToolPlan(
            retriever=DemoRetriever(),
            session_store=session,
            profile_store=profile,
            summary_store=summaries,
        ).run("How should this method connect vision and language?")

        print(f"context_action: {result.context.action.value}")
        print(f"resolved_paper_id: {result.context.resolved_paper_id}")
        print(f"needs_clarification: {result.needs_clarification}")
        print(f"retrieved_count: {result.retrieved_count}")
        print(f"memory_recall_count: {result.memory_recall_count}")
        print(f"session_written: {result.session_written}")
        print("tool_results:")
        for item in result.tool_results:
            print(f"- {item.call_id}: {item.tool_name} -> {item.status.value}")


if __name__ == "__main__":
    main()

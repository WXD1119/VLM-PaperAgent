from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from paper_agent.domain import ChunkKind
from paper_agent.graph.query import GraphQuery
from paper_agent.memory import (
    ContextGuard,
    ContextGuardDecision,
    GraphEntityResolver,
    SessionMemoryStore,
    SummaryMemoryStore,
    SummaryCursorStore,
    compress_episode_memory,
    UserProfileStore,
    build_summary_vector_store,
)
from paper_agent.tools.orchestrator import (
    ResourceAccess,
    ToolCall,
    ToolOrchestrator,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


class RetrieverLike(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[Any]: ...


class PaperQAToolPlanResult(BaseModel):
    query: str
    context: ContextGuardDecision
    tool_results: list[ToolResult] = Field(default_factory=list)
    retrieved_count: int = 0
    graph_candidate_count: int = 0
    memory_recall_count: int = 0
    session_written: bool = False
    needs_clarification: bool = False
    summary_written: bool = False
    summary_pending_events: int = 0


class PaperQAToolPlan:
    """论文问答的工具化回答前计划。

    该计划刻意停在 LLM 生成前，用于展示问答路径中的真实工具调用，
    同时保持测试快速且确定。
    """

    def __init__(
        self,
        *,
        retriever: RetrieverLike,
        session_store: SessionMemoryStore,
        profile_store: UserProfileStore,
        summary_store: SummaryMemoryStore | None = None,
        episode_store=None,
        summary_cursor_store: SummaryCursorStore | None = None,
        summary_window: int = 6,
        graph_query: GraphQuery | None = None,
        orchestrator: ToolOrchestrator | None = None,
    ) -> None:
        self.retriever = retriever
        self.session_store = session_store
        self.profile_store = profile_store
        self.summary_store = summary_store
        self.episode_store = episode_store
        self.summary_cursor_store = summary_cursor_store
        self.summary_window = summary_window
        self.graph_query = graph_query
        self.orchestrator = orchestrator or self._build_orchestrator()

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> PaperQAToolPlanResult:
        entity_resolver = GraphEntityResolver(self.graph_query.graph) if self.graph_query else None
        context = ContextGuard(entity_resolver).decide(
            query,
            explicit_paper_id=paper_id,
            session=self.session_store.load(),
            profile=self.profile_store.load(),
        )
        if context.needs_clarification:
            return PaperQAToolPlanResult(
                query=query,
                context=context,
                needs_clarification=True,
            )

        effective_paper_id = context.resolved_paper_id
        calls = [
            ToolCall(
                call_id="retrieve",
                tool_name="retrieve_chunks",
                args={
                    "query": query,
                    "top_k": top_k,
                    "paper_id": effective_paper_id,
                    "kind": kind,
                },
            ),
            ToolCall(
                call_id="graph",
                tool_name="query_graph_concepts",
                args={"query": query, "limit": top_k},
            ),
            ToolCall(
                call_id="memory_recall",
                tool_name="recall_summary_memory",
                args={"query": query, "top_k": min(top_k, 5)},
            ),
            ToolCall(
                call_id="session",
                tool_name="update_session",
                depends_on=("retrieve",),
                args={
                    "query": query,
                    "paper_id": effective_paper_id,
                },
            ),
        ]
        results = self.orchestrator.run_many(calls)
        by_id = {result.call_id: result for result in results}
        retrieve_output = by_id.get("retrieve").output if by_id.get("retrieve") else []
        graph_output = by_id.get("graph").output if by_id.get("graph") else []
        recall_output = (
            by_id.get("memory_recall").output if by_id.get("memory_recall") else []
        )
        session_result = by_id.get("session")
        compression = None
        if (
            session_result
            and session_result.status == ToolStatus.SUCCESS
            and self.episode_store is not None
            and self.summary_store is not None
            and self.summary_cursor_store is not None
        ):
            self.episode_store.log(
                "paper_qa_tool_plan",
                f"Processed paper QA query: {query}",
                {"paper_id": effective_paper_id, "query": query},
            )
            compression = compress_episode_memory(
                self.episode_store,
                self.summary_store,
                self.summary_cursor_store,
                window_size=self.summary_window,
                metadata={"source": "paper_qa_tool_plan"},
            )
        return PaperQAToolPlanResult(
            query=query,
            context=context,
            tool_results=results,
            retrieved_count=len(retrieve_output or []),
            graph_candidate_count=len(graph_output or []),
            memory_recall_count=len(recall_output or []),
            session_written=bool(session_result and session_result.status == ToolStatus.SUCCESS),
            summary_written=bool(compression and compression.written),
            summary_pending_events=(compression.pending_events if compression else 0),
        )

    def _build_orchestrator(self) -> ToolOrchestrator:
        return ToolOrchestrator(
            [
                ToolSpec(
                    name="retrieve_chunks",
                    handler=self._retrieve_chunks,
                    resources=frozenset({"paper_index"}),
                    access=ResourceAccess.READ,
                    max_retries=1,
                    description="Search paper chunks with the configured retriever.",
                ),
                ToolSpec(
                    name="query_graph_concepts",
                    handler=self._query_graph_concepts,
                    resources=frozenset({"paper_graph"}),
                    access=ResourceAccess.READ,
                    description="Find graph concept candidates related to the query.",
                ),
                ToolSpec(
                    name="update_session",
                    handler=self._update_session,
                    resources=frozenset({"session_memory"}),
                    access=ResourceAccess.WRITE,
                    max_retries=1,
                    description="Persist short-term QA task state.",
                ),
                ToolSpec(
                    name="recall_summary_memory",
                    handler=self._recall_summary_memory,
                    resources=frozenset({"summary_memory"}),
                    access=ResourceAccess.READ,
                    description="Recall relevant compressed episodic summaries.",
                ),
            ]
        )

    def _retrieve_chunks(self, args: dict[str, Any]) -> list[Any]:
        return self.retriever.search(
            args["query"],
            args["top_k"],
            paper_id=args.get("paper_id"),
            kind=args.get("kind"),
        )

    def _query_graph_concepts(self, args: dict[str, Any]) -> list[str]:
        if self.graph_query is None:
            return []
        return [
            node.label
            for node in self.graph_query.concept_candidates(args["query"], limit=args["limit"])
        ]

    def _update_session(self, args: dict[str, Any]) -> dict[str, str | None]:
        state = self.session_store.update(
            current_paper_id=args.get("paper_id"),
            last_query=args["query"],
            active_task="paper_qa_tool_plan",
        )
        return {
            "current_paper_id": state.current_paper_id,
            "last_query": state.last_query,
        }

    def _recall_summary_memory(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        if self.summary_store is None:
            return []
        summaries = self.summary_store.list()
        hits = build_summary_vector_store(summaries).search(args["query"], top_k=args["top_k"])
        return [
            {
                "summary_id": hit.summary.summary_id,
                "score": hit.score,
                "summary": hit.summary.summary,
                "paper_ids": hit.summary.paper_ids,
            }
            for hit in hits
        ]

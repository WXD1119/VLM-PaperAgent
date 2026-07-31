import json
import os
import queue
import threading
from datetime import timedelta
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from paper_agent.agents import (
    AnswerAgent,
    ComparisonBudget,
    ComparisonMatrix,
    DeterministicReviewPlanner,
    EvidenceBoundSectionWriter,
    ReviewSectionSemanticJudge,
    ReviewEvidence,
    ReviewExecutionResult,
    LiteratureReviewPlan,
    ReadingCompareServices,
    ReviewBudget,
    ReflectionAction,
    ReviewReflection,
    ReviewSectionDraft,
    ReviewSectionPlan,
    ReviewTemplate,
    CitationValidator,
    SemanticCitationJudge,
    build_evidence_pack,
    build_reading_compare_graph,
    build_review_writer_graph,
    ReviewWriterServices,
)
from paper_agent.domain import (
    AnswerBundle,
    ChunkBundle,
    ChunkKind,
    EvidencePack,
    GroundedAnswer,
    Paper,
    SemanticCitationReport,
)
from paper_agent.llm import OpenAICompatibleClient, RemoteStructuredClient, TransformersStructuredClient
from paper_agent.memory import (
    ContextGuard,
    ContextGuardDecision,
    ConversationTurn,
    MemorySummary,
    MySQLLongTermMemoryStore,
    MySQLUserProfileStore,
    RedisConversationWindowStore,
    RedisSessionMemoryStore,
    SessionMemoryStore,
    UserProfileStore,
    build_memory_summary,
    load_graph_entity_resolver,
)
from paper_agent.ingestion import IngestionStatus, IngestionTask, LocalIngestionTaskStore
from paper_agent.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore, StorageHealthReport, check_storage_health
from paper_agent.graph import (
    GraphQuery,
    GraphValidator,
    LocalGraphWorkspaceStore,
    build_paper_fragment,
    delta_from_fragment,
    read_graph_jsonl,
)
from paper_agent.graph.concepts import extract_concepts
from paper_agent.runtime import ResearchGraphRun, ResearchGraphServices, build_research_graph
from paper_agent.security import ActorContext, validate_query
from paper_agent.observability import JsonlTraceStore, TraceRecord, TraceRecorder
from paper_agent.orchestration import (
    AgentDispatch,
    ContextBudget,
    CorpusScope,
    GovernedWorkflowHooks,
    ResearchRouteDecision,
    ResearchRouteRequest,
    ResearchRouter,
    ScopeResolution,
    WorkflowGovernance,
)
from paper_agent.planning import (
    DeterministicResearchPlanner,
    LocalResearchPlanStore,
    MySQLResearchPlanStore,
    PaperQAResult,
    ResearchPlan,
    ResearchPlanExecutor,
    ResearchPlanStore,
)
from paper_agent.reviews import LocalReviewArtifactStore, ReviewArtifact, ReviewArtifactStore
from paper_agent.knowledge_base import LibraryUpdate, LocalKnowledgeBaseStore, RebuildJob, RebuildKind
from paper_agent.knowledge_base.coordinator import LibraryUpdateCoordinator
from paper_agent.knowledge_base.models import LibraryUpdatePhase
from paper_agent.temporary_workspace import (
    LocalTemporaryWorkspaceStore,
    TemporaryPaperRef,
    TemporaryPaperWorkspace,
    TemporaryWorkspaceIndexManager,
    TemporaryWorkspaceStore,
)
from paper_agent.tools import AuthorizedToolGateway, ToolRuntimePolicy

try:
    from fastapi import FastAPI, File, Header, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError:  # 可选依赖
    FastAPI = None  # type: ignore[assignment,misc]
    File = None  # type: ignore[assignment,misc]
    Header = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    UploadFile = None  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    paper_id: str | None = None
    kind: ChunkKind | None = None
    use_rerank: bool = True
    max_answer_attempts: int = Field(default=2, ge=1, le=5)
    use_context_guard: bool = True
    corpus_scope: CorpusScope = CorpusScope.AUTO
    temporary_workspace_id: str | None = Field(default=None, max_length=128)
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=20)
    web_expansion_confirmed: bool = False
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    # 仅由可信 API 边界写入，不能由请求体作为授权依据。
    user_id: str = Field(default="local-user", exclude=True)
    session_id: str = Field(default="local-session", exclude=True)


class AskResponse(BaseModel):
    bundle: AnswerBundle
    semantic_report: SemanticCitationReport | None = None
    semantic_gate_enabled: bool = False
    semantic_gate_passed: bool | None = None
    attempts: int = 1
    memory: MemorySummary | None = None
    context: ContextGuardDecision | None = None
    runtime: str = "langgraph"
    trace_id: str | None = None
    rewrite_count: int = 0
    refusal_kind: str | None = None
    route_intent: str | None = None
    route_confidence: float | None = None
    retrieval_policy: dict[str, object] = Field(default_factory=dict)
    corpus_scope: CorpusScope | None = None


class PaperSummary(BaseModel):
    paper_id: str
    title: str


class GraphConceptResponse(BaseModel):
    query: str
    resolved_concept: str | None = None
    requires_disambiguation: bool = False
    candidates: list[str] = Field(default_factory=list)
    papers: list[PaperSummary] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    chunks: list[dict[str, object]] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)


class WorkspacePromotionResponse(BaseModel):
    task_id: str
    workspace_path: str
    commit_id: str | None = None
    added_nodes: int
    added_edges: int
    valid_after_commit: bool
    status: str
    update_id: str | None = None


class ResearchPlanCreateRequest(BaseModel):
    """创建调研草案的请求；论文范围必须由用户显式确认。"""

    goal: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] = Field(min_length=2, max_length=12)


class ResearchPlanExecuteRequest(BaseModel):
    """执行草案前的显式确认，防止复杂任务静默消耗推理资源。"""

    confirm: bool = False


class ResearchDispatchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    mode: str | None = Field(default=None, max_length=128)
    active_paper_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=20)
    web_expansion_confirmed: bool = False


class ResearchRunRequest(ResearchDispatchRequest):
    """Run a router decision without exposing arbitrary tool selection to the client."""

    confirm: bool = False
    max_dimensions: int = Field(default=3, ge=1, le=8)
    max_sections: int = Field(default=4, ge=2, le=10)
    max_evidence_tasks: int = Field(default=24, ge=2, le=80)


class ResearchRunResponse(BaseModel):
    decision: ResearchRouteDecision
    answers: list[AskResponse] = Field(default_factory=list)
    comparison: ComparisonMatrix | None = None
    review: ReviewArtifact | None = None
    pending_dispatches: list[AgentDispatch] = Field(default_factory=list)


class ComparisonRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    paper_ids: list[str] = Field(min_length=2, max_length=12)
    max_dimensions: int = Field(default=3, ge=1, le=8)


class ReviewPlanRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=2_000)
    paper_ids: list[str] = Field(min_length=2, max_length=30)
    template: ReviewTemplate = ReviewTemplate.CONFERENCE_STYLE
    max_sections: int = Field(default=4, ge=2, le=10)
    max_evidence_tasks: int = Field(default=24, ge=2, le=80)


class ReviewRunRequest(ReviewPlanRequest):
    """Execution is explicitly confirmed because it can issue many paper QA calls."""

    confirm: bool = False


class TemporaryWorkspaceCreateRequest(BaseModel):
    ingestion_task_ids: list[str] = Field(min_length=1, max_length=20)
    ttl_hours: int = Field(default=24, ge=1, le=48)


class RebuildEnqueueRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=128)
    kind: RebuildKind
    target_parser_version: str | None = Field(default=None, max_length=128)
    target_embedding_model: str | None = Field(default=None, max_length=512)


class PaperQAService(Protocol):
    def ask(self, request: AskRequest) -> AskResponse: ...


LibraryIndexer = Callable[[ChunkBundle], None]


@runtime_checkable
class ProgressPaperQAService(PaperQAService, Protocol):
    """支持受控进度事件的问答服务契约。"""

    def ask_with_progress(
        self,
        request: AskRequest,
        on_event: Callable[[dict[str, object]], None],
    ) -> AskResponse: ...


@runtime_checkable
class TraceReadService(Protocol):
    def list_traces(self, *, user_id: str, limit: int = 50) -> list[TraceRecord]: ...

    def load_trace(self, trace_id: str, *, user_id: str) -> TraceRecord: ...


class GraphReadService(Protocol):
    def papers(self) -> list[PaperSummary]: ...

    def concept(self, query: str, limit: int) -> GraphConceptResponse: ...


class IngestionService(Protocol):
    def create(self, filename: str, content: bytes) -> IngestionTask: ...

    def list(self, limit: int = 20) -> list[IngestionTask]: ...

    def load(self, task_id: str) -> IngestionTask: ...

    def retry(self, task_id: str) -> IngestionTask: ...

    def mark_promoted(
        self,
        task_id: str,
        *,
        workspace_path: str,
        commit_id: str | None,
        message: str,
    ) -> IngestionTask: ...


class ApiResearchPlanService:
    """将受限计划执行器适配到已有论文问答服务，不直接暴露任意工具调用。"""

    def __init__(self, store: ResearchPlanStore, qa_service: PaperQAService, graph_reader: GraphReadService) -> None:
        self.store = store
        self.qa_service = qa_service
        self.graph_reader = graph_reader
        self.planner = DeterministicResearchPlanner()

    def create(self, *, user_id: str, goal: str, paper_ids: list[str]) -> ResearchPlan:
        normalized = list(dict.fromkeys(item.strip() for item in paper_ids if item.strip()))
        available = {paper.paper_id for paper in self.graph_reader.papers()}
        unknown = sorted(set(normalized) - available)
        if unknown:
            raise ValueError(f"不在当前论文图谱中的 paper_id: {', '.join(unknown)}")
        return self.store.create(
            self.planner.create_comparison_plan(user_id=user_id, goal=goal, paper_ids=normalized)
        )

    def list(self, *, user_id: str, limit: int) -> list[ResearchPlan]:
        return self.store.list(user_id, limit)

    def load(self, *, user_id: str, plan_id: str) -> ResearchPlan:
        return self.store.load(user_id, plan_id)

    def execute(self, *, user_id: str, session_id: str, plan_id: str, confirmed: bool) -> ResearchPlan:
        def ask_paper(question: str, paper_id: str) -> PaperQAResult:
            response = self.qa_service.ask(
                AskRequest(
                    query=question,
                    paper_id=paper_id,
                    user_id=user_id,
                    session_id=session_id,
                    top_k=5,
                )
            )
            return PaperQAResult(
                answer=response.bundle.answer.answer,
                evidence_chunk_ids=[item.chunk_id for item in response.bundle.evidence_pack.items],
                trace_id=response.trace_id,
                citation_valid=response.bundle.citation_validation.valid,
                abstained=response.bundle.answer.abstained,
            )

        return ResearchPlanExecutor(self.store, ask_paper).execute(
            user_id=user_id,
            plan_id=plan_id,
            confirmed=confirmed,
        )


class ApiReadingCompareService:
    """Run the bounded comparison subgraph through the existing scoped QA API."""

    def __init__(self, qa_service: PaperQAService, graph_reader: GraphReadService) -> None:
        self.qa_service = qa_service
        self.graph_reader = graph_reader

    def compare(
        self,
        *,
        query: str,
        paper_ids: list[str],
        user_id: str,
        session_id: str,
        max_dimensions: int,
    ) -> ComparisonMatrix:
        normalized = list(dict.fromkeys(item.strip() for item in paper_ids if item.strip()))
        available = {paper.paper_id for paper in self.graph_reader.papers()}
        unknown = sorted(set(normalized) - available)
        if unknown:
            raise ValueError(f"paper_id not available in the current library: {', '.join(unknown)}")

        def ask_paper(question: str, paper_id: str) -> PaperQAResult:
            response = self.qa_service.ask(
                AskRequest(
                    query=question,
                    paper_id=paper_id,
                    corpus_scope=CorpusScope.ACTIVE_PAPER_ONLY,
                    user_id=user_id,
                    session_id=session_id,
                    top_k=5,
                )
            )
            return PaperQAResult(
                answer=response.bundle.answer.answer,
                evidence_chunk_ids=[item.chunk_id for item in response.bundle.evidence_pack.items],
                trace_id=response.trace_id,
                citation_valid=response.bundle.citation_validation.valid,
                abstained=response.bundle.answer.abstained,
            )

        graph = build_reading_compare_graph(
            ReadingCompareServices(ask_paper=ask_paper)
        )
        state = graph.invoke(
            {
                "query": query,
                "paper_ids": normalized,
                "budget": ComparisonBudget(max_dimensions=max_dimensions),
            }
        )
        return state["matrix"]


class ApiReviewPlanService:
    """Create inspectable review plans; execution remains behind confirmation."""

    def __init__(self, graph_reader: GraphReadService) -> None:
        self.graph_reader = graph_reader
        self.planner = DeterministicReviewPlanner()

    def create(self, request: ReviewPlanRequest) -> LiteratureReviewPlan:
        normalized = list(dict.fromkeys(item.strip() for item in request.paper_ids if item.strip()))
        available = {paper.paper_id for paper in self.graph_reader.papers()}
        unknown = sorted(set(normalized) - available)
        if unknown:
            raise ValueError(f"paper_id not available in the current library: {', '.join(unknown)}")
        return self.planner.plan(
            request.topic,
            normalized,
            request.template,
            ReviewBudget(max_sections=request.max_sections, max_evidence_tasks=request.max_evidence_tasks),
        )


class ApiReviewWriterService(ApiReviewPlanService):
    """Adapt the bounded review LangGraph to the scoped paper QA service."""

    def __init__(
        self,
        qa_service: PaperQAService,
        graph_reader: GraphReadService,
        store: ReviewArtifactStore,
    ) -> None:
        super().__init__(graph_reader)
        self.qa_service = qa_service
        self.store = store

    def run(
        self,
        request: ReviewRunRequest,
        *,
        user_id: str,
        session_id: str,
    ) -> ReviewArtifact:
        if not request.confirm:
            raise ValueError("set confirm=true after reviewing the literature-review plan")
        # Validate the selection before any model/tool invocation.
        self.create(request)

        def ask_paper(question: str, paper_id: str) -> PaperQAResult:
            response = self.qa_service.ask(
                AskRequest(
                    query=question,
                    paper_id=paper_id,
                    corpus_scope=CorpusScope.ACTIVE_PAPER_ONLY,
                    user_id=user_id,
                    session_id=session_id,
                    top_k=5,
                )
            )
            return PaperQAResult(
                answer=response.bundle.answer.answer,
                evidence_chunk_ids=[item.chunk_id for item in response.bundle.evidence_pack.items],
                trace_id=response.trace_id,
                citation_valid=response.bundle.citation_validation.valid,
                abstained=response.bundle.answer.abstained,
            )

        def deterministic_write_section(section: ReviewSectionPlan, evidence: list[ReviewEvidence]) -> ReviewSectionDraft:
            usable = [item for item in evidence if item.citation_valid and not item.abstained]
            if not usable:
                content = "Insufficient citation-validated evidence was retrieved for this section."
            else:
                content = "\n\n".join(
                    f"[{item.paper_id}] {item.summary} (evidence: {', '.join(item.evidence_chunk_ids)})"
                    for item in usable
                )
            return ReviewSectionDraft(
                section_id=section.section_id,
                title=section.title,
                content=content,
                evidence_chunk_ids=[chunk_id for item in usable for chunk_id in item.evidence_chunk_ids],
                citation_valid=bool(usable),
            )

        # Production runtime uses a structured LLM writer; injected/test QA
        # services retain a deterministic writer without requiring a model.
        write_section = deterministic_write_section
        if isinstance(self.qa_service, LazyPaperQAService):
            write_section = EvidenceBoundSectionWriter(self.qa_service._llm_client()).write

        reflect = None
        if isinstance(self.qa_service, LazyPaperQAService) and self.qa_service.judge_url:
            semantic_judge = ReviewSectionSemanticJudge(self.qa_service._judge_client())

            def reflect(plan, drafts, evidence):
                reviews = [
                    semantic_judge.review(draft, [item for item in evidence if item.section_id == draft.section_id])
                    for draft in drafts
                ]
                non_passing = [item for item in reviews if item.suggested_action != ReflectionAction.PASS]
                if not non_passing:
                    return ReviewReflection(action=ReflectionAction.PASS, reasoning="all review sections passed semantic citation audit")
                # Evidence absence takes precedence over a rewrite; otherwise
                # preserve a bounded retrieve-more repair when the judge failed.
                action = next(
                    (item.suggested_action for item in non_passing if item.suggested_action == ReflectionAction.MARK_INSUFFICIENT_EVIDENCE),
                    next((item.suggested_action for item in non_passing if item.suggested_action == ReflectionAction.RETRIEVE_MORE), ReflectionAction.REWRITE_SECTION),
                )
                return ReviewReflection(
                    action=action,
                    affected_section_ids=[item.section_id for item in non_passing],
                    reasoning="; ".join(item.reasoning for item in non_passing),
                )

        graph = build_review_writer_graph(ReviewWriterServices(ask_paper=ask_paper, write_section=write_section, reflect=reflect))
        state = graph.invoke(
            {
                "topic": request.topic,
                "paper_ids": request.paper_ids,
                "template": request.template,
                "budget": ReviewBudget(max_sections=request.max_sections, max_evidence_tasks=request.max_evidence_tasks),
                "confirmed": True,
            }
        )
        return self.store.create(
            ReviewArtifact(
                user_id=user_id,
                execution=ReviewExecutionResult(
                    plan=state["plan"],
                    drafts=state.get("drafts", []),
                    reflection=state.get("reflection"),
                    waiting_confirmation=state.get("waiting_confirmation", False),
                    evidence_matrix=(state["evidence_matrix"].model_dump(mode="json") if state.get("evidence_matrix") else None),
                    validation=(state["validation"].model_dump(mode="json") if state.get("validation") else None),
                ),
            )
        )

    def load(self, *, user_id: str, review_id: str) -> ReviewArtifact:
        return self.store.load(user_id, review_id)

    def list(self, *, user_id: str, limit: int) -> list[ReviewArtifact]:
        return self.store.list(user_id, limit)


class LocalGraphReadService:
    """JSONL Paper KG 或有效 workspace 图谱的只读适配器。"""

    def __init__(
        self,
        graph_path: str | Path = "artifacts/graph",
        workspace_path: str | Path | None = None,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.workspace_path = Path(workspace_path) if workspace_path else None

    def papers(self) -> list[PaperSummary]:
        return [
            PaperSummary(
                paper_id=str(node.properties.get("paper_id", node.node_id.removeprefix("paper:"))),
                title=node.label,
            )
            for node in GraphQuery(self._graph()).papers()
        ]

    def concept(self, query: str, limit: int) -> GraphConceptResponse:
        neighborhood = GraphQuery(self._graph()).concept_neighborhood(query, limit=limit)
        return GraphConceptResponse(
            query=query,
            resolved_concept=neighborhood.concept.label if neighborhood.concept else None,
            requires_disambiguation=neighborhood.concept is None and bool(neighborhood.candidates),
            candidates=[node.label for node in neighborhood.candidates],
            papers=[
                PaperSummary(
                    paper_id=str(node.properties.get("paper_id", node.node_id.removeprefix("paper:"))),
                    title=node.label,
                )
                for node in neighborhood.papers
            ],
            sections=[node.label for node in neighborhood.sections],
            chunks=[
                {
                    "chunk_id": str(node.properties.get("chunk_id", node.node_id)),
                    "paper_id": str(node.properties.get("paper_id", "")),
                    "pages": node.properties.get("pages", []),
                    "section": node.properties.get("section_path", []),
                    "preview": str(node.properties.get("content_preview", node.label)),
                }
                for node in neighborhood.chunks
            ],
            related_concepts=[node.label for node in neighborhood.related_concepts],
        )

    def _graph(self):
        if self.workspace_path is not None:
            return LocalGraphWorkspaceStore(self.workspace_path).load_effective_graph()
        return read_graph_jsonl(self.graph_path)


class LazyPaperQAService:
    """检索、生成与可选语义门控的延迟初始化服务封装。"""

    def __init__(self) -> None:
        self.chunks_path = os.getenv("PAPER_AGENT_CHUNKS", "artifacts/papers")
        self.db_path = Path(os.getenv("PAPER_AGENT_CHROMA_DB", "artifacts/chroma"))
        self.collection = os.getenv("PAPER_AGENT_CHROMA_COLLECTION", "paper_chunks_bge_m3")
        self.embedding_model = os.getenv("PAPER_AGENT_EMBEDDING_MODEL", "artifacts/models/bge-m3")
        self.reranker_model = os.getenv(
            "PAPER_AGENT_RERANKER_MODEL",
            "artifacts/models/bge-reranker-v2-m3",
        )
        self.generator_model = os.getenv(
            "PAPER_AGENT_LLM_MODEL",
            "artifacts/models/Qwen3-VL-8B-Instruct",
        )
        self.provider = os.getenv("PAPER_AGENT_PROVIDER", "local")
        self.base_url = os.getenv("PAPER_AGENT_LLM_BASE_URL", "https://api.deepseek.com")
        self.api_key_env = os.getenv("PAPER_AGENT_API_KEY_ENV", "DEEPSEEK_API_KEY")
        self.device = os.getenv("PAPER_AGENT_DEVICE") or None
        self.llm_device = os.getenv("PAPER_AGENT_LLM_DEVICE") or None
        self.offline = os.getenv("PAPER_AGENT_OFFLINE", "1") not in {"0", "false", "False"}
        self.rrf_k = int(os.getenv("PAPER_AGENT_RRF_K", "60"))
        self.judge_url = os.getenv("PAPER_AGENT_JUDGE_URL") or None
        self.session_memory_path = os.getenv(
            "PAPER_AGENT_SESSION_MEMORY",
            "artifacts/memory/session.json",
        )
        self.profile_memory_path = os.getenv(
            "PAPER_AGENT_PROFILE_MEMORY",
            "artifacts/memory/user_profile.json",
        )
        self.graph_path = os.getenv("PAPER_AGENT_GRAPH", "artifacts/graph")
        self.graph_workspace_path = os.getenv("PAPER_AGENT_GRAPH_WORKSPACE") or None
        self.memory_backend = os.getenv("PAPER_AGENT_MEMORY_BACKEND", "file")
        self.redis_url = os.getenv("REDIS_URL")
        self.mysql_url = os.getenv("MYSQL_URL")
        self.trace_store = JsonlTraceStore(os.getenv("PAPER_AGENT_TRACE_PATH", "artifacts/traces/traces.jsonl"))
        self.temporary_workspace_root = os.getenv("PAPER_AGENT_TEMPORARY_WORKSPACE_ROOT", "artifacts/temporary_workspaces")
        self.temporary_index_root = os.getenv("PAPER_AGENT_TEMPORARY_INDEX_ROOT", "artifacts/temporary_indexes")

        self._sparse = None
        self._dense = None
        self._reranker = None
        self._client = None
        self._judge = None
        self._entity_resolver = None
        self._entity_resolver_loaded = False
        self._mysql_memory = None
        self._temporary_index = None
        self.runtime = os.getenv("PAPER_AGENT_RUNTIME", "langgraph")

    def ask(self, request: AskRequest) -> AskResponse:
        return self._ask(request)

    def ask_with_progress(
        self,
        request: AskRequest,
        on_event: Callable[[dict[str, object]], None],
    ) -> AskResponse:
        """执行问答并发送可展示的节点状态，不暴露模型原始推理。"""

        return self._ask(request, on_event=on_event)

    def _ask(
        self,
        request: AskRequest,
        *,
        on_event: Callable[[dict[str, object]], None] | None = None,
    ) -> AskResponse:
        validate_query(request.query)
        recorder = TraceRecorder(
            query=request.query,
            runtime=self.runtime,
            user_id=request.user_id,
            session_id=request.session_id,
            event_listener=(
                lambda event: on_event({"type": "node", **event.model_dump()})
                if on_event
                else None
            ),
        )
        memory_warnings: list[str] = []
        try:
            if self.runtime == "legacy":
                response = recorder.run("legacy_workflow", lambda: self._ask_legacy(request))
            else:
                response = self._ask_langgraph(request, recorder, memory_warnings)
            try:
                recorder.run("persist_memory", lambda: self._persist_conversation(request, response))
            except Exception as exc:
                # 记忆服务不可用时保留问答结果，并将降级原因写进 Trace 与响应。
                warning = f"memory persistence degraded: {type(exc).__name__}"
                memory_warnings.append(warning)
                recorder.record_event("memory_degraded", status="degraded", elapsed_ms=0, attributes={"reason": warning})
            if memory_warnings:
                response = response.model_copy(
                    update={
                        "memory": MemorySummary(
                            status="answer returned; distributed memory degraded to file/no-write mode",
                            recommendation="inspect trace and restore Redis/MySQL before relying on conversation memory",
                        )
                    }
                )
            trace = recorder.complete(
                status="success",
                attributes={
                    "citation_valid": response.bundle.citation_validation.valid,
                    "semantic_gate_passed": response.semantic_gate_passed,
                    "attempts": response.attempts,
                    "rewrite_count": response.rewrite_count,
                    "refusal_kind": response.refusal_kind,
                    "evidence_count": len(response.bundle.evidence_pack.items),
                    "abstained": response.bundle.answer.abstained,
                    "intent": response.route_intent,
                    "route_confidence": response.route_confidence,
                    "retrieval_policy": response.retrieval_policy,
                },
            )
            if on_event:
                on_event({"type": "completed", "trace_id": trace.trace_id, "status": trace.status})
            return response.model_copy(update={"trace_id": trace.trace_id})
        except Exception as exc:
            trace = recorder.complete(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            if on_event:
                on_event({"type": "error", "error": trace.error or "问答执行失败"})
            raise
        finally:
            try:
                self.trace_store.append(trace)
            except Exception:
                # 追踪故障不能反向破坏已经完成的问答主流程。
                pass

    def list_traces(self, *, user_id: str, limit: int = 50) -> list[TraceRecord]:
        """按用户返回脱敏后的近期运行轨迹。"""

        return self.trace_store.list(limit=limit, user_id=user_id)

    def load_trace(self, trace_id: str, *, user_id: str) -> TraceRecord:
        """按用户读取一条运行轨迹，避免跨用户查看调试信息。"""

        return self.trace_store.load(trace_id, user_id=user_id)

    def _ask_langgraph(
        self,
        request: AskRequest,
        recorder: TraceRecorder,
        memory_warnings: list[str],
    ) -> AskResponse:
        request = self._resolve_temporary_workspace_request(request)
        actor = ActorContext(user_id=request.user_id, session_id=request.session_id)
        gateway = AuthorizedToolGateway(
            actor,
            policies={
                "retrieve_evidence": ToolRuntimePolicy(timeout_s=45.0, max_retries=1, resources=frozenset({"retrieval"})),
                "generate_answer": ToolRuntimePolicy(timeout_s=180.0, resources=frozenset({"generator"})),
                "semantic_judge": ToolRuntimePolicy(timeout_s=180.0, resources=frozenset({"judge"})),
            },
        )

        def resolve_context(query: str, paper_id: str | None) -> ContextGuardDecision:
            if not request.use_context_guard:
                return ContextGuardDecision(query=query, action="proceed", resolved_paper_id=paper_id)
            try:
                session = self._session_store(request).load()
                profile = self._profile_store(request).load()
            except Exception as exc:
                # 上下文记忆不可用时回退为空文件记忆，避免影响证据问答主链路。
                memory_warnings.append(f"memory context degraded: {type(exc).__name__}")
                session = SessionMemoryStore(self.session_memory_path).load()
                profile = UserProfileStore(self.profile_memory_path).load()
            return ContextGuard(self._context_entity_resolver()).decide(
                query,
                explicit_paper_id=paper_id,
                session=session,
                profile=profile,
            )

        def retrieve(
            query: str,
            top_k: int,
            paper_id: str | None,
            kind: object | None,
            use_rerank: bool,
        ) -> list[object]:
            scoped_request = request.model_copy(
                update={"paper_id": paper_id, "kind": kind, "use_rerank": use_rerank}
            )
            if scoped_request.corpus_scope == CorpusScope.TEMPORARY_WORKSPACE:
                if not scoped_request.temporary_workspace_id:
                    raise ValueError("temporary_workspace scope requires temporary_workspace_id")
                return self._temporary_retriever(scoped_request).search(
                    user_id=scoped_request.user_id,
                    session_id=scoped_request.session_id,
                    workspace_id=scoped_request.temporary_workspace_id,
                    query=query,
                    top_k=top_k,
                    paper_id=paper_id,
                    kind=kind,
                )
            return self._retriever(scoped_request).search(query, top_k, paper_id=paper_id, kind=kind)

        graph = build_research_graph(
            ResearchGraphServices(
                resolve_context=resolve_context,
                retrieve=retrieve,
                build_evidence=build_evidence_pack,
                answer=AnswerAgent(self._llm_client()).answer,
                judge=(self._judge_client().evaluate if self.judge_url else None),
                graph_candidates=self._graph_candidate_papers,
                run_tool=gateway.run,
                hooks=GovernedWorkflowHooks(
                    WorkflowGovernance(
                        actor=actor,
                        # The graph replaces this initial scope after
                        # ContextGuard resolves the actual corpus boundary.
                        scope=ScopeResolution(scope=request.corpus_scope),
                        context_budget=request.context_budget,
                        web_expansion_confirmed=request.web_expansion_confirmed,
                    )
                ),
            )
        )
        state = graph.invoke(
            {
                "query": request.query,
                "paper_id": request.paper_id,
                "corpus_scope": request.corpus_scope,
                "selected_paper_ids": request.selected_paper_ids,
                "web_expansion_confirmed": request.web_expansion_confirmed,
                "context_budget": request.context_budget,
                "kind": request.kind,
                "top_k": request.top_k,
                "use_rerank": request.use_rerank,
                "max_attempts": request.max_answer_attempts,
                "trace_recorder": recorder,
            }
        )
        run = ResearchGraphRun(state)
        bundle = AnswerBundle(
            evidence_pack=run.evidence_pack,
            answer=run.answer,
            citation_validation=run.citation_validation,
            generator_model=self.generator_model,
        )
        return AskResponse(
            bundle=bundle,
            semantic_report=run.semantic_report,
            semantic_gate_enabled=run.semantic_gate_enabled,
            semantic_gate_passed=run.semantic_gate_passed,
            attempts=run.attempts,
            memory=build_memory_summary(bundle, run.semantic_report),
            context=run.context,
            runtime="langgraph",
            rewrite_count=run.rewrite_count,
            refusal_kind=run.refusal_kind,
            route_intent=run.route_plan.intent.value if run.route_plan else None,
            route_confidence=run.route_plan.confidence if run.route_plan else None,
            retrieval_policy=(
                {
                    "evidence_types": [item.value for item in run.route_plan.required_evidence_types],
                    "preferred_sections": run.route_plan.preferred_sections,
                    "use_rerank": run.route_plan.use_rerank,
                    "use_graph": run.route_plan.use_graph,
                    "use_judge": run.route_plan.use_judge,
                    "sub_question_count": len(run.evidence_plan.sub_questions) if run.evidence_plan else 0,
                    "minimum_evidence_count": run.evidence_plan.minimum_evidence_count if run.evidence_plan else 0,
                }
                if run.route_plan
                else {}
            ),
            corpus_scope=(run.scope_resolution.scope if getattr(run, "scope_resolution", None) else None),
        )

    def _ask_legacy(self, request: AskRequest) -> AskResponse:
        context_decision = None
        effective_paper_id = request.paper_id
        if request.use_context_guard:
            context_decision = ContextGuard(self._context_entity_resolver()).decide(
                request.query,
                explicit_paper_id=request.paper_id,
                session=self._session_store(request).load(),
                profile=self._profile_store(request).load(),
            )
            if context_decision.needs_clarification:
                return self._clarification_response(request, context_decision)
            effective_paper_id = context_decision.resolved_paper_id
        retriever = self._retriever(request)
        hits = retriever.search(
            request.query,
            request.top_k,
            paper_id=effective_paper_id,
            kind=request.kind,
        )
        pack = build_evidence_pack(request.query, hits)
        agent = AnswerAgent(self._llm_client())
        judge = self._judge_client() if self.judge_url else None
        bundle, semantic_report, attempts, gate_enabled = self._answer_with_gate(
            agent=agent,
            pack=pack,
            judge=judge,
            max_attempts=request.max_answer_attempts,
        )
        bundle = bundle.model_copy(update={"generator_model": self.generator_model})
        gate_passed = None
        if gate_enabled and not bundle.answer.abstained:
            gate_passed = bool(semantic_report and semantic_report.all_supported)
        return AskResponse(
            bundle=bundle,
            semantic_report=semantic_report,
            semantic_gate_enabled=gate_enabled,
            semantic_gate_passed=gate_passed,
            attempts=attempts,
            memory=build_memory_summary(bundle, semantic_report),
            context=context_decision,
            runtime="legacy",
        )

    def _context_entity_resolver(self):
        if not self._entity_resolver_loaded:
            self._entity_resolver = load_graph_entity_resolver(
                graph_path=self.graph_path,
                workspace_path=self.graph_workspace_path,
            )
            self._entity_resolver_loaded = True
        return self._entity_resolver

    def _session_store(self, request: AskRequest):
        if self.memory_backend != "distributed":
            return SessionMemoryStore(self.session_memory_path)
        if not self.redis_url:
            raise RuntimeError("REDIS_URL is required when PAPER_AGENT_MEMORY_BACKEND=distributed")
        return RedisSessionMemoryStore(
            self.redis_url,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    def _profile_store(self, request: AskRequest):
        if self.memory_backend != "distributed":
            return UserProfileStore(self.profile_memory_path)
        if not self.mysql_url:
            raise RuntimeError("MYSQL_URL is required when PAPER_AGENT_MEMORY_BACKEND=distributed")
        if self._mysql_memory is None:
            self._mysql_memory = MySQLLongTermMemoryStore(self.mysql_url)
        return MySQLUserProfileStore(self._mysql_memory, request.user_id)

    def _persist_conversation(self, request: AskRequest, response: AskResponse) -> None:
        """持久化用户会话记忆，但不触碰 Paper KG。

        文件模式保留旧版仅回答工件的行为；分布式模式将受限原始窗口写入 Redis，
        将可审计的持久副本写入 MySQL。
        """

        if self.memory_backend != "distributed":
            return
        if self._mysql_memory is None:
            # _profile_store 会校验 MYSQL_URL 并初始化仓储。
            self._profile_store(request)
        assert self._mysql_memory is not None
        window = RedisConversationWindowStore(
            self.redis_url or "",
            user_id=request.user_id,
            session_id=request.session_id,
        )
        turns = [
            ConversationTurn(role="user", content=request.query),
            ConversationTurn(
                role="assistant",
                content=response.bundle.answer.answer,
                metadata={
                    "abstained": response.bundle.answer.abstained,
                    "paper_id": response.context.resolved_paper_id if response.context else request.paper_id,
                    "runtime": response.runtime,
                },
            ),
        ]
        for turn in turns:
            window.append(turn)
            self._mysql_memory.append_turn(request.user_id, request.session_id, turn)
        self._session_store(request).update(
            last_query=request.query,
            current_paper_id=(response.context.resolved_paper_id if response.context else request.paper_id),
        )

    @staticmethod
    def _clarification_response(
        request: AskRequest,
        context_decision: ContextGuardDecision,
    ) -> AskResponse:
        safe_answer = GroundedAnswer(
            answer=context_decision.clarification_question
            or "I need clarification before searching the paper corpus.",
            abstained=True,
            abstention_reason="The query depends on missing conversation context.",
        )
        pack = EvidencePack(query=request.query, items=[])
        validation = CitationValidator().validate(safe_answer, pack)
        bundle = AnswerBundle(
            evidence_pack=pack,
            answer=safe_answer,
            citation_validation=validation,
            generator_model="",
        )
        return AskResponse(
            bundle=bundle,
            attempts=0,
            memory=build_memory_summary(bundle),
            context=context_decision,
        )

    def _retriever(self, request: AskRequest):
        if self._sparse is None:
            chunks = chunks_from_bundles(load_chunk_bundles(self.chunks_path))
            self._sparse = BM25Index(chunks)
        if self._dense is None:
            encoder = SentenceTransformerEncoder(
                self.embedding_model,
                device=self.device,
                local_files_only=self.offline,
            )
            self._dense = ChromaVectorStore(self.db_path, self.collection, encoder)
        hybrid = HybridRetriever(self._sparse, self._dense, self.rrf_k, request.candidate_k)
        if not request.use_rerank:
            return hybrid
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.reranker_model,
                device=self.device,
                local_files_only=self.offline,
            )
        return RerankedRetriever(hybrid, self._reranker, request.candidate_k)

    def _resolve_temporary_workspace_request(self, request: AskRequest) -> AskRequest:
        if request.corpus_scope != CorpusScope.TEMPORARY_WORKSPACE:
            return request
        if not request.temporary_workspace_id:
            raise ValueError("temporary_workspace scope requires temporary_workspace_id")
        workspace = self._temporary_workspace_store().load(
            user_id=request.user_id,
            session_id=request.session_id,
            workspace_id=request.temporary_workspace_id,
        )
        paper_ids = [item.paper_id for item in workspace.papers]
        if request.paper_id is None:
            if len(paper_ids) != 1:
                raise ValueError("choose paper_id when a temporary workspace contains multiple papers")
            return request.model_copy(update={"paper_id": paper_ids[0]})
        if request.paper_id not in paper_ids:
            raise PermissionError("paper_id is outside the temporary workspace")
        return request

    def _temporary_retriever(self, request: AskRequest):
        if self._temporary_index is None:
            encoder = SentenceTransformerEncoder(
                self.embedding_model,
                device=self.device,
                local_files_only=self.offline,
            )
            self._temporary_index = TemporaryWorkspaceIndexManager(
                self._temporary_workspace_store(),
                encoder,
                root=self.temporary_index_root,
                rrf_k=self.rrf_k,
                candidate_k=request.candidate_k,
            )
        return self._temporary_index

    def _temporary_workspace_store(self) -> LocalTemporaryWorkspaceStore:
        return LocalTemporaryWorkspaceStore(
            self.temporary_workspace_root,
            on_delete=lambda workspace: TemporaryWorkspaceIndexManager.remove_persisted(
                self.temporary_index_root, workspace.workspace_id
            ),
        )

    def _graph_candidate_papers(self, query: str, limit: int) -> list[str]:
        graph = (
            LocalGraphWorkspaceStore(self.graph_workspace_path).load_effective_graph()
            if self.graph_workspace_path
            else read_graph_jsonl(self.graph_path)
        )
        graph_query = GraphQuery(graph)
        candidates: list[str] = []
        # Extracting compact concept phrases avoids treating the whole natural
        # language question as a graph-node key.
        for mention in extract_concepts(query):
            neighborhood = graph_query.concept_neighborhood(mention.label, limit=limit)
            candidates.extend(str(item.properties.get("paper_id", "")) for item in neighborhood.papers)
            candidates.extend(str(item.properties.get("paper_id", "")) for item in neighborhood.chunks)
        if not candidates:
            candidates.extend(
                str(item.properties.get("paper_id", ""))
                for item in graph_query.search_nodes(query, limit=limit)
            )
        return [paper_id for paper_id in dict.fromkeys(candidates) if paper_id][:limit]

    def _llm_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "local":
            self._client = TransformersStructuredClient(self.generator_model, device=self.llm_device)
            return self._client
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {self.api_key_env} is required")
        self._client = OpenAICompatibleClient(self.generator_model, api_key, self.base_url)
        return self._client

    def _judge_client(self):
        if self._judge is None:
            self._judge = SemanticCitationJudge(RemoteStructuredClient(self.judge_url))
        return self._judge

    @staticmethod
    def _answer_with_gate(
        *,
        agent: AnswerAgent,
        pack,
        judge: SemanticCitationJudge | None,
        max_attempts: int,
    ) -> tuple[AnswerBundle, SemanticCitationReport | None, int, bool]:
        feedback: str | None = None
        last_report: SemanticCitationReport | None = None
        for attempt in range(1, max_attempts + 1):
            answer, validation = agent.answer(pack, feedback=feedback)
            bundle = AnswerBundle(
                evidence_pack=pack,
                answer=answer,
                citation_validation=validation,
                generator_model="",
            )
            if judge is None or answer.abstained:
                return bundle, None, attempt, bool(judge)
            report = judge.evaluate(answer, pack)
            last_report = report
            if report.all_supported:
                return bundle, report, attempt, True
            feedback = "\n".join(
                f"Claim {item.claim_index} was judged {item.verdict.value}: "
                f"{item.reasoning_summary}"
                for item in report.assessments
                if item.verdict.value != "supported"
            )

        safe_answer = GroundedAnswer(
            answer="I cannot provide a fully citation-supported answer from the retrieved evidence.",
            abstained=True,
            abstention_reason="Semantic citation validation failed after all attempts.",
        )
        validation = CitationValidator().validate(safe_answer, pack)
        return (
            AnswerBundle(
                evidence_pack=pack,
                answer=safe_answer,
                citation_validation=validation,
                generator_model="",
            ),
            last_report,
            max_attempts,
            True,
        )


def create_app(
    service: PaperQAService | None = None,
    graph_service: GraphReadService | None = None,
    ingestion_service: IngestionService | None = None,
    research_plan_store: ResearchPlanStore | None = None,
    review_artifact_store: ReviewArtifactStore | None = None,
    temporary_workspace_store: TemporaryWorkspaceStore | None = None,
    knowledge_base_store: LocalKnowledgeBaseStore | None = None,
    library_indexer: LibraryIndexer | None = None,
):
    if FastAPI is None:
        raise RuntimeError("Install the api extra: pip install -e '.[api]'")
    app = FastAPI(title="VLM-PaperAgent API", version="0.1.0")
    qa_service = service or LazyPaperQAService()
    graph_reader = graph_service or LocalGraphReadService(
        graph_path=os.getenv("PAPER_AGENT_GRAPH", "artifacts/graph"),
        workspace_path=os.getenv("PAPER_AGENT_GRAPH_WORKSPACE") or None,
    )
    ingestion = ingestion_service or LocalIngestionTaskStore(
        os.getenv("PAPER_AGENT_INGESTION_ROOT", "artifacts/ingestion")
    )
    if research_plan_store is not None:
        plan_store = research_plan_store
    elif os.getenv("PAPER_AGENT_MEMORY_BACKEND", "file") == "distributed" and os.getenv("MYSQL_URL"):
        plan_store = MySQLResearchPlanStore(os.environ["MYSQL_URL"])
    else:
        plan_store = LocalResearchPlanStore(
            os.getenv("PAPER_AGENT_RESEARCH_PLAN_ROOT", "artifacts/research_plans")
        )
    research_plans = ApiResearchPlanService(plan_store, qa_service, graph_reader)
    reading_compare = ApiReadingCompareService(qa_service, graph_reader)
    review_plans = ApiReviewPlanService(graph_reader)
    review_store = review_artifact_store or LocalReviewArtifactStore(
        os.getenv("PAPER_AGENT_REVIEW_ROOT", "artifacts/reviews")
    )
    review_writer = ApiReviewWriterService(qa_service, graph_reader, review_store)
    knowledge_base = knowledge_base_store or LocalKnowledgeBaseStore(
        os.getenv("PAPER_AGENT_KNOWLEDGE_BASE_ROOT", "artifacts/knowledge_base")
    )
    def publish_library_index(chunks: ChunkBundle) -> None:
        if library_indexer is not None:
            library_indexer(chunks)
            return
        encoder = SentenceTransformerEncoder(
            os.getenv("PAPER_AGENT_EMBEDDING_MODEL", "artifacts/models/bge-m3"),
            device=os.getenv("PAPER_AGENT_DEVICE") or None,
            local_files_only=os.getenv("PAPER_AGENT_OFFLINE", "1") not in {"0", "false", "False"},
        )
        store = ChromaVectorStore(
            os.getenv("PAPER_AGENT_CHROMA_DB", "artifacts/chroma"),
            os.getenv("PAPER_AGENT_CHROMA_COLLECTION", "paper_chunks_bge_m3"),
            encoder,
        )
        store.upsert(chunks.children)
    temporary_index_root = os.getenv("PAPER_AGENT_TEMPORARY_INDEX_ROOT", "artifacts/temporary_indexes")
    temporary_workspaces = temporary_workspace_store or LocalTemporaryWorkspaceStore(
        os.getenv("PAPER_AGENT_TEMPORARY_WORKSPACE_ROOT", "artifacts/temporary_workspaces"),
        on_delete=lambda workspace: TemporaryWorkspaceIndexManager.remove_persisted(
            temporary_index_root, workspace.workspace_id
        ),
    )

    def actor_from_headers(
        x_user_id: str | None,
        x_session_id: str | None,
    ) -> ActorContext:
        """Accept identity headers only in explicitly trusted-proxy mode.

        A raw HTTP header is not authentication. Production must set this mode only
        behind a gateway that validates the user identity (for example JWT/OIDC).
        """

        mode = os.getenv("PAPER_AGENT_AUTH_MODE", "local")
        if mode == "trusted-header":
            if not x_user_id or not x_session_id:
                raise HTTPException(status_code=401, detail="authenticated user and session headers required")
            return ActorContext(user_id=x_user_id, session_id=x_session_id)
        return ActorContext(user_id="local-user", session_id=x_session_id or "local-session")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/storage", response_model=StorageHealthReport)
    def storage_health() -> StorageHealthReport:
        """返回三类存储的无数据健康状态，用于部署验收与故障排查。"""

        return check_storage_health(
            redis_url=os.getenv("REDIS_URL"),
            mysql_url=os.getenv("MYSQL_URL"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
        )

    @app.get("/traces", response_model=list[TraceRecord])
    def list_traces(
        limit: int = 20,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> list[TraceRecord]:
        if not isinstance(qa_service, TraceReadService):
            raise HTTPException(status_code=501, detail="trace endpoint requires LazyPaperQAService")
        actor = actor_from_headers(x_user_id, x_session_id)
        return qa_service.list_traces(user_id=actor.user_id, limit=max(1, min(limit, 100)))

    @app.get("/traces/{trace_id}", response_model=TraceRecord)
    def get_trace(
        trace_id: str,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> TraceRecord:
        if not isinstance(qa_service, TraceReadService):
            raise HTTPException(status_code=501, detail="trace endpoint requires LazyPaperQAService")
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            return qa_service.load_trace(trace_id, user_id=actor.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="trace not found") from exc

    @app.get("/papers", response_model=list[PaperSummary])
    def papers() -> list[PaperSummary]:
        try:
            return graph_reader.papers()
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/temporary-workspaces", response_model=TemporaryPaperWorkspace, status_code=201)
    def create_temporary_workspace(
        request: TemporaryWorkspaceCreateRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> TemporaryPaperWorkspace:
        actor = actor_from_headers(x_user_id, x_session_id)
        references: list[TemporaryPaperRef] = []
        try:
            for task_id in list(dict.fromkeys(request.ingestion_task_ids)):
                task = ingestion.load(task_id)
                if task.status != IngestionStatus.SUCCEEDED or not task.paper_id or not task.paper_path or not task.chunks_path:
                    raise ValueError(f"ingestion task is not ready for temporary reading: {task_id}")
                if task.workspace_commit_id:
                    raise ValueError(f"promoted ingestion task cannot enter a temporary workspace: {task_id}")
                references.append(
                    TemporaryPaperRef(
                        paper_id=task.paper_id,
                        ingestion_task_id=task.task_id,
                        title=task.filename,
                        paper_path=task.paper_path,
                        chunks_path=task.chunks_path,
                    )
                )
            return temporary_workspaces.create(
                user_id=actor.user_id,
                session_id=actor.session_id,
                papers=references,
                ttl=timedelta(hours=request.ttl_hours),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="ingestion task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/temporary-workspaces/{workspace_id}", response_model=TemporaryPaperWorkspace)
    def get_temporary_workspace(
        workspace_id: str,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> TemporaryPaperWorkspace:
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            return temporary_workspaces.load(user_id=actor.user_id, session_id=actor.session_id, workspace_id=workspace_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="temporary workspace not found") from exc

    @app.delete("/temporary-workspaces/{workspace_id}", status_code=204)
    def delete_temporary_workspace(
        workspace_id: str,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> None:
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            deleted = temporary_workspaces.delete(user_id=actor.user_id, session_id=actor.session_id, workspace_id=workspace_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="temporary workspace not found") from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="temporary workspace not found")
        # The temporary vector collection is stored only below this dedicated
        # root; use the same validated ID to remove its persisted index.

    @app.get("/research/plans", response_model=list[ResearchPlan])
    def list_research_plans(
        limit: int = 20,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> list[ResearchPlan]:
        actor = actor_from_headers(x_user_id, x_session_id)
        return research_plans.list(user_id=actor.user_id, limit=max(1, min(limit, 100)))

    @app.post("/research/dispatch", response_model=ResearchRouteDecision)
    def dispatch_research(request: ResearchDispatchRequest) -> ResearchRouteDecision:
        return ResearchRouter().route(
            ResearchRouteRequest(
                query=request.query,
                mode=request.mode,
                active_paper_id=request.active_paper_id,
                selected_paper_ids=request.selected_paper_ids,
                web_expansion_confirmed=request.web_expansion_confirmed,
            )
        )

    @app.post("/research/run", response_model=ResearchRunResponse)
    def run_research(
        request: ResearchRunRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ResearchRunResponse:
        """Execute only router-approved subgraphs in their declared dependency order."""

        actor = actor_from_headers(x_user_id, x_session_id)
        decision = ResearchRouter().route(
            ResearchRouteRequest(
                query=request.query,
                mode=request.mode,
                active_paper_id=request.active_paper_id,
                selected_paper_ids=request.selected_paper_ids,
                web_expansion_confirmed=request.web_expansion_confirmed,
            )
        )
        if decision.needs_clarification:
            return ResearchRunResponse(decision=decision, pending_dispatches=decision.dispatches)

        answers: list[AskResponse] = []
        comparison: ComparisonMatrix | None = None
        review: ReviewArtifact | None = None
        pending: list[AgentDispatch] = []
        completed_modes: set[str] = set()
        for dispatch in decision.dispatches:
            if any(item not in completed_modes for item in dispatch.depends_on):
                pending.append(dispatch)
                continue
            if dispatch.needs_confirmation and not request.confirm:
                pending.append(dispatch)
                continue
            if dispatch.mode == "manage_papers":
                # Ingestion requires a file upload and remains a separate explicit API.
                pending.append(dispatch)
                continue
            if dispatch.mode in {"active_paper_explain", "library_related_work"}:
                try:
                    answers.append(
                        qa_service.ask(
                            AskRequest(
                                query=request.query,
                                paper_id=request.active_paper_id if dispatch.mode == "active_paper_explain" else None,
                                corpus_scope=dispatch.corpus_scope,
                                selected_paper_ids=request.selected_paper_ids,
                                web_expansion_confirmed=request.web_expansion_confirmed,
                                user_id=actor.user_id,
                                session_id=actor.session_id,
                            )
                        )
                    )
                    completed_modes.add(dispatch.mode)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                continue
            if dispatch.mode == "multi_paper_compare":
                try:
                    comparison = reading_compare.compare(
                        query=request.query,
                        paper_ids=request.selected_paper_ids,
                        user_id=actor.user_id,
                        session_id=actor.session_id,
                        max_dimensions=request.max_dimensions,
                    )
                    completed_modes.add(dispatch.mode)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                continue
            if dispatch.mode == "review_write":
                if not request.confirm:
                    pending.append(dispatch)
                    continue
                try:
                    review = review_writer.run(
                        ReviewRunRequest(
                            topic=request.query,
                            paper_ids=request.selected_paper_ids,
                            max_sections=request.max_sections,
                            max_evidence_tasks=request.max_evidence_tasks,
                            confirm=True,
                        ),
                        user_id=actor.user_id,
                        session_id=actor.session_id,
                    )
                    completed_modes.add(dispatch.mode)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                continue
            pending.append(dispatch)
        return ResearchRunResponse(
            decision=decision,
            answers=answers,
            comparison=comparison,
            review=review,
            pending_dispatches=pending,
        )

    @app.post("/research/compare", response_model=ComparisonMatrix)
    def compare_papers(
        request: ComparisonRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ComparisonMatrix:
        try:
            actor = actor_from_headers(x_user_id, x_session_id)
            return reading_compare.compare(
                query=request.query,
                paper_ids=request.paper_ids,
                user_id=actor.user_id,
                session_id=actor.session_id,
                max_dimensions=request.max_dimensions,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/research/reviews/plan", response_model=LiteratureReviewPlan)
    def create_review_plan(request: ReviewPlanRequest) -> LiteratureReviewPlan:
        try:
            return review_plans.create(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/research/reviews", response_model=list[ReviewArtifact])
    def list_reviews(
        limit: int = 20,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> list[ReviewArtifact]:
        actor = actor_from_headers(x_user_id, x_session_id)
        return review_writer.list(user_id=actor.user_id, limit=max(1, min(limit, 100)))

    @app.get("/research/reviews/{review_id}", response_model=ReviewArtifact)
    def get_review(
        review_id: str,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ReviewArtifact:
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            return review_writer.load(user_id=actor.user_id, review_id=review_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="review not found") from exc

    @app.post("/research/reviews/run", response_model=ReviewArtifact)
    def run_review(
        request: ReviewRunRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ReviewExecutionResult:
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            return review_writer.run(request, user_id=actor.user_id, session_id=actor.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/research/plans", response_model=ResearchPlan, status_code=201)
    def create_research_plan(
        request: ResearchPlanCreateRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ResearchPlan:
        try:
            actor = actor_from_headers(x_user_id, x_session_id)
            return research_plans.create(user_id=actor.user_id, goal=request.goal, paper_ids=request.paper_ids)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/research/plans/{plan_id}", response_model=ResearchPlan)
    def get_research_plan(
        plan_id: str,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ResearchPlan:
        actor = actor_from_headers(x_user_id, x_session_id)
        try:
            return research_plans.load(user_id=actor.user_id, plan_id=plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="research plan not found") from exc

    @app.post("/research/plans/{plan_id}/execute", response_model=ResearchPlan)
    def execute_research_plan(
        plan_id: str,
        request: ResearchPlanExecuteRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> ResearchPlan:
        try:
            actor = actor_from_headers(x_user_id, x_session_id)
            return research_plans.execute(
                user_id=actor.user_id,
                session_id=actor.session_id,
                plan_id=plan_id,
                confirmed=request.confirm,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="research plan not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/graph/concepts", response_model=GraphConceptResponse)
    def graph_concepts(q: str, limit: int = 8) -> GraphConceptResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="q must not be empty")
        try:
            return graph_reader.concept(q.strip(), max(1, min(limit, 20)))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/knowledge-base/updates", response_model=list[LibraryUpdate])
    def list_library_updates(limit: int = 20) -> list[LibraryUpdate]:
        return knowledge_base.list_updates(max(1, min(limit, 100)))

    @app.get("/knowledge-base/papers/{paper_id}/versions", response_model=list[LibraryUpdate])
    def list_paper_versions(paper_id: str) -> list[LibraryUpdate]:
        return [item for item in knowledge_base.list_updates(limit=10_000) if item.paper_id == paper_id]

    @app.get("/knowledge-base/rebuilds", response_model=list[RebuildJob])
    def list_rebuilds(limit: int = 20) -> list[RebuildJob]:
        return knowledge_base.list_rebuilds(max(1, min(limit, 100)))

    @app.post("/knowledge-base/rebuilds", response_model=RebuildJob, status_code=202)
    def enqueue_rebuild(request: RebuildEnqueueRequest) -> RebuildJob:
        if request.kind == RebuildKind.REPARSE and not request.target_parser_version:
            raise HTTPException(status_code=422, detail="target_parser_version is required for reparse")
        if request.kind == RebuildKind.REEMBED and not request.target_embedding_model:
            raise HTTPException(status_code=422, detail="target_embedding_model is required for reembed")
        return knowledge_base.enqueue_rebuild(
            RebuildJob(
                paper_id=request.paper_id,
                kind=request.kind,
                target_parser_version=request.target_parser_version,
                target_embedding_model=request.target_embedding_model,
            )
        )

    @app.get("/ingestion/tasks", response_model=list[IngestionTask])
    def list_ingestion_tasks(limit: int = 20) -> list[IngestionTask]:
        try:
            return ingestion.list(max(1, min(limit, 100)))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/ingestion/tasks/{task_id}", response_model=IngestionTask)
    def get_ingestion_task(task_id: str) -> IngestionTask:
        try:
            return ingestion.load(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ingestion/tasks", response_model=IngestionTask, status_code=201)
    async def create_ingestion_task(file: UploadFile = File(...)) -> IngestionTask:
        try:
            if not file.filename:
                raise ValueError("uploaded file must have a filename")
            return ingestion.create(file.filename, await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ingestion/tasks/{task_id}/retry", response_model=IngestionTask)
    def retry_ingestion_task(task_id: str) -> IngestionTask:
        try:
            return ingestion.retry(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/ingestion/tasks/{task_id}/promote",
        response_model=WorkspacePromotionResponse,
    )
    def promote_ingestion_task(task_id: str) -> WorkspacePromotionResponse:
        try:
            task = ingestion.load(task_id)
            if task.status.value != "succeeded" or not task.paper_path or not task.chunks_path:
                raise ValueError("only a successfully parsed task can be promoted")
            workspace_path = os.getenv("PAPER_AGENT_GRAPH_WORKSPACE")
            if not workspace_path:
                raise ValueError("PAPER_AGENT_GRAPH_WORKSPACE must be configured")
            paper = Paper.model_validate_json(Path(task.paper_path).read_text(encoding="utf-8"))
            chunks = ChunkBundle.model_validate_json(Path(task.chunks_path).read_text(encoding="utf-8"))
            workspace = LocalGraphWorkspaceStore(workspace_path)
            delta = delta_from_fragment(
                workspace.load_effective_graph(),
                build_paper_fragment(paper, chunks),
            )
            parser_version = paper.elements[0].parser_version if paper.elements else "unknown"
            update = LibraryUpdate(
                paper_id=paper.paper_id,
                content_sha256=paper.sha256,
                parser_version=parser_version,
                embedding_model=os.getenv("PAPER_AGENT_EMBEDDING_MODEL", "artifacts/models/bge-m3"),
                ingestion_task_id=task_id,
            )
            commit_id: str | None = None

            def validate(_update: LibraryUpdate) -> None:
                report = GraphValidator().validate(workspace.preview_delta(delta))
                if not report.valid:
                    raise ValueError("workspace promotion would make graph invalid: " + "; ".join(report.errors))

            def index(_update: LibraryUpdate) -> None:
                if not chunks.children:
                    raise ValueError("cannot promote a paper with no retrieval chunks")
                publish_library_index(chunks)

            def graph_action(_update: LibraryUpdate) -> str | None:
                nonlocal commit_id
                if not delta.added_nodes and not delta.added_edges:
                    return None
                commit = workspace.commit_delta(
                    delta,
                    author_id=os.getenv("PAPER_AGENT_WORKSPACE_AUTHOR", "web-user"),
                    message=f"add uploaded paper {paper.paper_id}",
                )
                commit_id = commit.commit_id
                return commit_id

            def metadata(_update: LibraryUpdate) -> None:
                ingestion.mark_promoted(
                    task_id,
                    workspace_path=workspace_path,
                    commit_id=commit_id,
                    message=(f"promoted paper through update {_update.update_id}"),
                )

            result = LibraryUpdateCoordinator(
                knowledge_base,
                {
                    LibraryUpdatePhase.VALIDATE: validate,
                    LibraryUpdatePhase.INDEX: index,
                    LibraryUpdatePhase.GRAPH: graph_action,
                    LibraryUpdatePhase.METADATA: metadata,
                },
            ).run(update)
            return WorkspacePromotionResponse(
                task_id=task_id,
                workspace_path=workspace_path,
                commit_id=result.graph_commit_id,
                added_nodes=len(delta.added_nodes),
                added_edges=len(delta.added_edges),
                valid_after_commit=True,
                status="already_visible" if not delta.added_nodes and not delta.added_edges else "promoted",
                update_id=result.update_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ask", response_model=AskResponse)
    def ask(
        request: AskRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ) -> AskResponse:
        try:
            actor = actor_from_headers(x_user_id, x_session_id)
            scoped_request = request.model_copy(
                update={"user_id": actor.user_id, "session_id": actor.session_id}
            )
            response = qa_service.ask(scoped_request)
            if response.memory is None:
                response = response.model_copy(
                    update={
                        "memory": build_memory_summary(
                            response.bundle,
                            response.semantic_report,
                        )
                    }
                )
            if response.context is None:
                response = response.model_copy(
                    update={
                        "context": ContextGuard().decide(
                            scoped_request.query,
                            explicit_paper_id=scoped_request.paper_id,
                        )
                    }
                )
            return response
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/ask/stream")
    def ask_stream(
        request: AskRequest,
        x_user_id: str | None = Header(default=None),
        x_session_id: str | None = Header(default=None),
    ):
        """以 SSE 返回受控执行状态，最终事件才携带正式回答。"""

        if not isinstance(qa_service, ProgressPaperQAService):
            raise HTTPException(status_code=501, detail="streaming requires LazyPaperQAService")
        actor = actor_from_headers(x_user_id, x_session_id)
        scoped_request = request.model_copy(
            update={"user_id": actor.user_id, "session_id": actor.session_id}
        )
        events: queue.Queue[dict[str, object] | None] = queue.Queue()

        def publish(event: dict[str, object]) -> None:
            events.put(event)

        def run() -> None:
            try:
                publish({"type": "started", "message": "已接收问题，正在启动受控工作流"})
                response = qa_service.ask_with_progress(scoped_request, publish)
                publish({"type": "answer", "response": response.model_dump(mode="json")})
            except Exception as exc:
                publish({"type": "error", "error": str(exc)[:500]})
            finally:
                events.put(None)

        def stream() -> Iterator[str]:
            worker = threading.Thread(target=run, daemon=True, name="paper-agent-sse")
            worker.start()
            while True:
                event = events.get()
                if event is None:
                    break
                yield f"event: {event.get('type', 'progress')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app() if FastAPI is not None else None

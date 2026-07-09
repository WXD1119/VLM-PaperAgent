import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from paper_agent.agents import AnswerAgent, CitationValidator, SemanticCitationJudge, build_evidence_pack
from paper_agent.domain import AnswerBundle, ChunkKind, GroundedAnswer, SemanticCitationReport
from paper_agent.llm import OpenAICompatibleClient, RemoteStructuredClient, TransformersStructuredClient
from paper_agent.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # optional dependency
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    paper_id: str | None = None
    kind: ChunkKind | None = None
    use_rerank: bool = True
    max_answer_attempts: int = Field(default=2, ge=1, le=5)


class AskResponse(BaseModel):
    bundle: AnswerBundle
    semantic_report: SemanticCitationReport | None = None
    semantic_gate_enabled: bool = False
    semantic_gate_passed: bool | None = None
    attempts: int = 1


class PaperQAService(Protocol):
    def ask(self, request: AskRequest) -> AskResponse: ...


class LazyPaperQAService:
    """Lazy service wrapper around retrieval, generation, and optional semantic gate."""

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

        self._sparse = None
        self._dense = None
        self._reranker = None
        self._client = None
        self._judge = None

    def ask(self, request: AskRequest) -> AskResponse:
        retriever = self._retriever(request)
        hits = retriever.search(
            request.query,
            request.top_k,
            paper_id=request.paper_id,
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


def create_app(service: PaperQAService | None = None):
    if FastAPI is None:
        raise RuntimeError("Install the api extra: pip install -e '.[api]'")
    app = FastAPI(title="VLM-PaperAgent API", version="0.1.0")
    qa_service = service or LazyPaperQAService()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            return qa_service.ask(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app() if FastAPI is not None else None

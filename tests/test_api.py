import pytest
import pydantic

if not hasattr(pydantic, "model_validator"):
    pytest.skip("API tests require pydantic v2", allow_module_level=True)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from paper_agent.api.main import (
    AskRequest,
    AskResponse,
    GraphConceptResponse,
    PaperSummary,
    create_app,
)
from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    Paper,
    RetrievalChunk,
    ChunkBundle,
)
from paper_agent.graph import GraphDocument, LocalGraphWorkspaceStore, write_graph_jsonl
from paper_agent.ingestion import IngestionStatus, IngestionTask
from paper_agent.reviews import LocalReviewArtifactStore
from paper_agent.temporary_workspace import LocalTemporaryWorkspaceStore
from paper_agent.knowledge_base import LocalKnowledgeBaseStore


class FakeQAService:
    def ask(self, request: AskRequest) -> AskResponse:
        bundle = AnswerBundle(
            evidence_pack=EvidencePack(
                query=request.query,
                items=[
                    EvidenceItem(
                        evidence_id="E1",
                        chunk_id="chunk-1",
                        paper_id="paper-1",
                        kind=ChunkKind.TEXT,
                        pages=[1],
                        section_path=["Method"],
                        content="Evidence text",
                    )
                ],
            ),
            answer=GroundedAnswer(
                answer="Answer text",
                claims=[AnswerClaim(text="Claim text", evidence_ids=["E1"])],
            ),
            citation_validation=CitationValidation(
                valid=True,
                claim_count=1,
                cited_claim_count=1,
            ),
            generator_model="fake-model",
        )
        return AskResponse(
            bundle=bundle,
            semantic_gate_enabled=False,
            attempts=1,
        )


class FakeProgressQAService(FakeQAService):
    def ask_with_progress(self, request: AskRequest, on_event) -> AskResponse:
        on_event({"type": "node", "name": "retrieve_evidence", "status": "running", "elapsed_ms": 0})
        on_event({"type": "node", "name": "retrieve_evidence", "status": "success", "elapsed_ms": 8})
        return self.ask(request)


class FakeGraphService:
    def papers(self) -> list[PaperSummary]:
        return [PaperSummary(paper_id="paper-1", title="Example Paper")]

    def concept(self, query: str, limit: int) -> GraphConceptResponse:
        assert query == "Q-Former"
        assert limit == 8
        return GraphConceptResponse(
            query=query,
            resolved_concept="Q-Former",
            candidates=["Q-Former"],
            papers=[PaperSummary(paper_id="paper-1", title="Example Paper")],
            chunks=[
                {
                    "chunk_id": "chunk-1",
                    "paper_id": "paper-1",
                    "pages": [2],
                    "section": ["Method"],
                    "preview": "Evidence text",
                }
            ],
        )


class TwoPaperGraphService(FakeGraphService):
    def papers(self) -> list[PaperSummary]:
        return [
            PaperSummary(paper_id="paper-1", title="Example Paper One"),
            PaperSummary(paper_id="paper-2", title="Example Paper Two"),
        ]


class FakeIngestionService:
    def __init__(self) -> None:
        self.task = IngestionTask(
            task_id="ing_123",
            filename="paper.pdf",
            source_path="artifacts/ingestion/uploads/ing_123.pdf",
        )

    def create(self, filename: str, content: bytes) -> IngestionTask:
        assert filename == "paper.pdf"
        assert content == b"%PDF-test"
        return self.task

    def list(self, limit: int = 20) -> list[IngestionTask]:
        return [self.task]

    def load(self, task_id: str) -> IngestionTask:
        assert task_id == self.task.task_id
        return self.task

    def retry(self, task_id: str) -> IngestionTask:
        return self.load(task_id)

    def mark_promoted(
        self,
        task_id: str,
        *,
        workspace_path: str,
        commit_id: str | None,
        message: str,
    ) -> IngestionTask:
        return self.load(task_id)


def test_health_endpoint():
    client = TestClient(create_app(FakeQAService()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer_bundle():
    client = TestClient(create_app(FakeQAService()))
    response = client.post("/ask", json={"query": "What training objective does CLIP use?"})
    assert response.status_code == 200
    body = response.json()
    assert body["bundle"]["answer"]["answer"] == "Answer text"
    assert body["bundle"]["answer"]["claims"][0]["evidence_ids"] == ["E1"]
    assert body["bundle"]["citation_validation"]["valid"] is True
    assert body["semantic_gate_enabled"] is False
    assert body["memory"]["paper_kg_written"] is False
    assert body["memory"]["recommendation"] == "ask user before long-term archiving"
    assert body["memory"]["requires_user_confirmation"] is True
    assert body["context"]["action"] == "proceed"
    assert body["context"]["needs_clarification"] is False


def test_ask_stream_returns_progress_before_final_answer():
    client = TestClient(create_app(FakeProgressQAService()))
    response = client.post("/ask/stream", json={"query": "What is Q-Former?"})
    assert response.status_code == 200
    assert "event: node" in response.text
    assert '"name": "retrieve_evidence"' in response.text
    assert "event: answer" in response.text
    assert "Answer text" in response.text


def test_graph_endpoints_expose_papers_and_concept_evidence():
    client = TestClient(create_app(FakeQAService(), FakeGraphService()))
    papers = client.get("/papers")
    assert papers.status_code == 200
    assert papers.json() == [{"paper_id": "paper-1", "title": "Example Paper"}]

    concept = client.get("/graph/concepts?q=Q-Former")
    assert concept.status_code == 200
    body = concept.json()
    assert body["resolved_concept"] == "Q-Former"
    assert body["papers"][0]["paper_id"] == "paper-1"
    assert body["chunks"][0]["chunk_id"] == "chunk-1"


def test_confirmed_review_run_executes_and_persists_scoped_paper_qa_workflow(tmp_path):
    client = TestClient(
        create_app(
            FakeQAService(),
            TwoPaperGraphService(),
            review_artifact_store=LocalReviewArtifactStore(tmp_path / "reviews"),
        )
    )
    rejected = client.post(
        "/research/reviews/run",
        json={"topic": "vision-language learning", "paper_ids": ["paper-1", "paper-2"]},
    )
    assert rejected.status_code == 422

    response = client.post(
        "/research/reviews/run",
        json={
            "topic": "vision-language learning",
            "paper_ids": ["paper-1", "paper-2"],
            "max_sections": 2,
            "max_evidence_tasks": 4,
            "confirm": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_id"].startswith("review_")
    assert body["execution"]["waiting_confirmation"] is False
    assert len(body["execution"]["drafts"]) == 2
    assert body["execution"]["drafts"][0]["citation_valid"] is True

    listed = client.get("/research/reviews")
    assert listed.status_code == 200
    assert listed.json()[0]["review_id"] == body["review_id"]
    loaded = client.get(f"/research/reviews/{body['review_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["execution"]["plan"]["topic"] == "vision-language learning"


def test_router_run_executes_only_approved_reading_workflow():
    client = TestClient(create_app(FakeQAService(), FakeGraphService()))
    response = client.post(
        "/research/run",
        json={"query": "Explain the method", "mode": "reading", "active_paper_id": "paper-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [item["bundle"]["answer"]["answer"] for item in body["answers"]] == ["Answer text"]
    assert body["pending_dispatches"] == []


def test_router_run_keeps_review_pending_until_confirmed():
    client = TestClient(create_app(FakeQAService(), TwoPaperGraphService()))
    response = client.post(
        "/research/run",
        json={"query": "vision-language learning", "mode": "review", "selected_paper_ids": ["paper-1", "paper-2"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answers"] == []
    assert body["pending_dispatches"][0]["mode"] == "review_write"


def test_ingestion_endpoints_create_and_list_tasks():
    client = TestClient(create_app(FakeQAService(), FakeGraphService(), FakeIngestionService()))
    created = client.post(
        "/ingestion/tasks",
        files={"file": ("paper.pdf", b"%PDF-test", "application/pdf")},
    )
    assert created.status_code == 201
    assert created.json()["task_id"] == "ing_123"

    listed = client.get("/ingestion/tasks")
    assert listed.status_code == 200
    assert listed.json()[0]["filename"] == "paper.pdf"


def test_temporary_workspace_accepts_only_ready_unpromoted_ingestion(tmp_path):
    ingestion = FakeIngestionService()
    ingestion.task.status = IngestionStatus.SUCCEEDED
    ingestion.task.paper_id = "temporary-paper"
    ingestion.task.paper_path = str(tmp_path / "paper.json")
    ingestion.task.chunks_path = str(tmp_path / "chunks.json")
    client = TestClient(
        create_app(
            FakeQAService(),
            FakeGraphService(),
            ingestion,
            temporary_workspace_store=LocalTemporaryWorkspaceStore(tmp_path / "temporary"),
        )
    )
    created = client.post("/temporary-workspaces", json={"ingestion_task_ids": ["ing_123"], "ttl_hours": 2})
    assert created.status_code == 201
    workspace_id = created.json()["workspace_id"]
    assert created.json()["papers"][0]["paper_id"] == "temporary-paper"
    assert client.get(f"/temporary-workspaces/{workspace_id}").status_code == 200
    assert client.delete(f"/temporary-workspaces/{workspace_id}").status_code == 204
    assert client.get(f"/temporary-workspaces/{workspace_id}").status_code == 404


def test_successful_ingestion_can_be_promoted_to_configured_workspace(tmp_path, monkeypatch):
    base_graph = tmp_path / "base_graph"
    write_graph_jsonl(GraphDocument(), base_graph)
    workspace_path = tmp_path / "workspace"
    LocalGraphWorkspaceStore(workspace_path).create_fork(
        base_graph_path=base_graph,
        workspace_id="ws-test",
        owner_id="tester",
    )
    paper_path = tmp_path / "paper.json"
    chunks_path = tmp_path / "chunks.json"
    paper = Paper(paper_id="paper-1", title="Test Paper", source_path="test.pdf", sha256="abc")
    chunks = ChunkBundle(
        paper_id="paper-1",
        parents=[],
        children=[
            RetrievalChunk(
                chunk_id="chunk-1",
                parent_chunk_id="parent-1",
                paper_id="paper-1",
                kind=ChunkKind.TEXT,
                content="Q-Former uses learnable queries.",
                element_ids=["element-1"],
                pages=[1],
                section_path=["Method"],
            )
        ],
    )
    paper_path.write_text(paper.model_dump_json(), encoding="utf-8")
    chunks_path.write_text(chunks.model_dump_json(), encoding="utf-8")
    ingestion = FakeIngestionService()
    ingestion.task.status = IngestionStatus.SUCCEEDED
    ingestion.task.paper_path = str(paper_path)
    ingestion.task.chunks_path = str(chunks_path)
    monkeypatch.setenv("PAPER_AGENT_GRAPH_WORKSPACE", str(workspace_path))

    client = TestClient(
        create_app(
            FakeQAService(),
            FakeGraphService(),
            ingestion,
            knowledge_base_store=LocalKnowledgeBaseStore(tmp_path / "knowledge_base"),
            library_indexer=lambda _chunks: None,
        )
    )
    response = client.post("/ingestion/tasks/ing_123/promote")
    assert response.status_code == 200
    assert response.json()["status"] == "promoted"
    assert response.json()["update_id"].startswith("update_")
    versions = client.get("/knowledge-base/papers/paper-1/versions")
    assert versions.status_code == 200
    assert versions.json()[0]["update_id"] == response.json()["update_id"]
    queued = client.post(
        "/knowledge-base/rebuilds",
        json={"paper_id": "paper-1", "kind": "reembed", "target_embedding_model": "bge-m3-v2"},
    )
    assert queued.status_code == 202
    assert client.get("/knowledge-base/rebuilds").json()[0]["job_id"] == queued.json()["job_id"]
    assert response.json()["added_nodes"] > 0

import pytest
import pydantic

if not hasattr(pydantic, "model_validator"):
    pytest.skip("API tests require pydantic v2", allow_module_level=True)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from paper_agent.api.main import AskRequest, AskResponse, create_app
from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
)


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

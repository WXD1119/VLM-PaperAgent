import pytest

from paper_agent.discovery import CandidatePaperDiscovery, DiscoveryRequest
from paper_agent.orchestration import CorpusScope, GovernanceViolation, WorkflowGovernance, WorkflowLimits, resolve_scope
from paper_agent.security import ActorContext


class FakeMetadataClient:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        return self.payload


def _discovery(payload):
    governance = WorkflowGovernance(
        actor=ActorContext(user_id="u1", session_id="s1", scopes=frozenset({"web:discover"})),
        scope=resolve_scope(CorpusScope.WEB_EXPANSION, web_expansion_confirmed=True),
        web_expansion_confirmed=True,
        limits=WorkflowLimits(max_network_calls=1),
    )
    client = FakeMetadataClient(payload)
    return CandidatePaperDiscovery(client, governance=governance), client


def test_discovery_requires_explicit_confirmation_and_web_scope_before_http():
    discovery, client = _discovery({"message": {"items": []}})
    with pytest.raises(GovernanceViolation, match="confirmed web_expansion"):
        discovery.discover(DiscoveryRequest(query="retrieval augmented generation", corpus_scope=CorpusScope.LIBRARY_ONLY, confirmed=True))
    with pytest.raises(GovernanceViolation, match="confirmed web_expansion"):
        discovery.discover(DiscoveryRequest(query="retrieval augmented generation", corpus_scope=CorpusScope.WEB_EXPANSION, confirmed=False))
    assert client.urls == []


def test_discovery_validates_deduplicates_metadata_and_never_fetches_candidate_urls():
    payload = {
        "message": {
            "items": [
                {"title": ["RAG Paper"], "DOI": "https://doi.org/10.1000/ABC", "author": [{"given": "Ada", "family": "Lovelace"}], "published-online": {"date-parts": [[2024]]}},
                {"title": ["RAG paper duplicate"], "DOI": "10.1000/abc"},
                {"title": ["Missing DOI"]},
                {"title": ["Second Paper"], "DOI": "10.1000/second", "container-title": ["Journal"]},
            ]
        }
    }
    discovery, client = _discovery(payload)
    result = discovery.discover(DiscoveryRequest(query="  retrieval  augmented generation ", corpus_scope=CorpusScope.WEB_EXPANSION, confirmed=True))
    assert result.query == "retrieval augmented generation"
    assert [candidate.doi for candidate in result.candidates] == ["10.1000/abc", "10.1000/second"]
    assert result.candidates[0].authors == ["Ada Lovelace"]
    assert result.rejected_records == 1
    assert len(client.urls) == 1
    assert client.urls[0].startswith("https://api.crossref.org/works?")
    assert "doi.org/10.1000" not in client.urls[0]


def test_discovery_is_subject_to_network_quota():
    discovery, _ = _discovery({"message": {"items": []}})
    request = DiscoveryRequest(query="paper agents", corpus_scope=CorpusScope.WEB_EXPANSION, confirmed=True)
    discovery.discover(request)
    with pytest.raises(GovernanceViolation, match="network call budget exhausted"):
        discovery.discover(request)

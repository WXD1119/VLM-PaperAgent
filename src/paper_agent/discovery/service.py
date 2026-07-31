"""A narrow adapter for finding importable paper *metadata* on the web.

It does not fetch landing pages, PDFs, or supplementary files.  Keeping this
boundary small prevents an LLM from turning a discovery request into an
unbounded network/download workflow.
"""

from __future__ import annotations

import re
from typing import Any, Protocol
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, HttpUrl, field_validator

from paper_agent.orchestration import CorpusScope, GovernanceViolation, WorkflowGovernance


class MetadataHttpClient(Protocol):
    """Injectable, metadata-only HTTP capability used by a discovery provider."""

    def get_json(self, url: str) -> dict[str, Any]: ...


class DiscoveryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    corpus_scope: CorpusScope
    confirmed: bool = False
    limit: int = Field(default=10, ge=1, le=25)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("query must contain at least two non-space characters")
        return normalized


class CandidatePaper(BaseModel):
    """Validated, review-only metadata; never contains a full-text URL."""

    title: str = Field(min_length=1, max_length=1_000)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1600, le=2200)
    doi: str | None = Field(default=None, max_length=300)
    arxiv_id: str | None = Field(default=None, max_length=80)
    venue: str | None = Field(default=None, max_length=500)
    abstract: str | None = Field(default=None, max_length=8_000)
    source: str = Field(min_length=1, max_length=64)
    metadata_url: HttpUrl

    @field_validator("doi")
    @classmethod
    def normalize_doi(cls, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().lower()
        normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
        if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", normalized, flags=re.IGNORECASE):
            raise ValueError("doi is invalid")
        return normalized

    @field_validator("arxiv_id")
    @classmethod
    def normalize_arxiv_id(cls, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().lower().removeprefix("arxiv:")
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", normalized):
            raise ValueError("arxiv_id is invalid")
        return normalized


class DiscoveryResult(BaseModel):
    query: str
    candidates: list[CandidatePaper] = Field(default_factory=list)
    provider: str
    rejected_records: int = Field(default=0, ge=0)
    requires_import_confirmation: bool = True


class CandidatePaperDiscovery:
    """Discover deduplicated candidates from Crossref's metadata API.

    The caller must provide governance configured with ``web:discover``.  The
    separate request checks make misuse fail closed even if this class is called
    outside a LangGraph node.
    """

    CROSSREF_WORKS_URL = "https://api.crossref.org/works"

    def __init__(self, client: MetadataHttpClient, *, governance: WorkflowGovernance) -> None:
        self.client = client
        self.governance = governance

    def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        if request.corpus_scope != CorpusScope.WEB_EXPANSION or not request.confirmed:
            raise GovernanceViolation("candidate discovery requires confirmed web_expansion scope")

        # This is the only network-capable operation in this package.  It is an
        # allow-listed metadata endpoint; no candidate-provided URL is fetched.
        self.governance.before_tool("web_discover", network=True)
        url = f"{self.CROSSREF_WORKS_URL}?query={quote_plus(request.query)}&rows={request.limit}"
        payload = self.client.get_json(url)
        records = payload.get("message", {}).get("items", [])
        if not isinstance(records, list):
            raise ValueError("Crossref metadata response has invalid items")

        candidates: list[CandidatePaper] = []
        rejected = 0
        seen: set[str] = set()
        for record in records:
            try:
                candidate = self._from_crossref(record)
            except (TypeError, ValueError):
                rejected += 1
                continue
            identity = _dedupe_key(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)
            if len(candidates) >= request.limit:
                break
        return DiscoveryResult(query=request.query, candidates=candidates, provider="crossref", rejected_records=rejected)

    @staticmethod
    def _from_crossref(record: Any) -> CandidatePaper:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        titles = record.get("title")
        title = titles[0].strip() if isinstance(titles, list) and titles and isinstance(titles[0], str) else ""
        doi = record.get("DOI")
        if not title or not isinstance(doi, str):
            raise ValueError("record lacks required title or DOI")
        authors = [
            " ".join(part for part in (author.get("given"), author.get("family")) if isinstance(part, str)).strip()
            for author in record.get("author", [])
            if isinstance(author, dict)
        ]
        year = _first_year(record.get("published-print")) or _first_year(record.get("published-online"))
        venues = record.get("container-title")
        venue = venues[0].strip() if isinstance(venues, list) and venues and isinstance(venues[0], str) else None
        abstract = record.get("abstract")
        metadata_url = f"https://doi.org/{doi.strip()}"
        return CandidatePaper(
            title=title,
            authors=[author for author in authors if author],
            year=year,
            doi=doi,
            venue=venue,
            abstract=abstract if isinstance(abstract, str) else None,
            source="crossref",
            metadata_url=metadata_url,
        )


def _first_year(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list) or not parts[0]:
        return None
    year = parts[0][0]
    return year if isinstance(year, int) and 1600 <= year <= 2200 else None


def _dedupe_key(candidate: CandidatePaper) -> str:
    if candidate.doi:
        return f"doi:{candidate.doi}"
    if candidate.arxiv_id:
        return f"arxiv:{candidate.arxiv_id}"
    normalized_title = re.sub(r"\W+", "", candidate.title.casefold())
    return f"title:{normalized_title}:{candidate.year or ''}"

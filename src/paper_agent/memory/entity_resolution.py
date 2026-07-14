from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from paper_agent.graph import EdgeType, GraphDocument, GraphQuery, NodeType
from paper_agent.graph import LocalGraphWorkspaceStore, read_graph_jsonl


class EntityResolutionStatus(StrEnum):
    NO_MATCH = "no_match"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"


class PaperEntityCandidate(BaseModel):
    paper_id: str
    title: str
    score: float
    matched_entities: list[str] = Field(default_factory=list)


class EntityResolution(BaseModel):
    status: EntityResolutionStatus
    matched_entity: str | None = None
    resolved_paper_id: str | None = None
    candidates: list[PaperEntityCandidate] = Field(default_factory=list)
    reason: str = ""


class EntityResolver(Protocol):
    def resolve(self, query: str) -> EntityResolution: ...


class GraphEntityResolver:
    """Resolve explicit query entities to papers using the Paper KG.

    Paper-title matches are authoritative. Concept matches are scored by how often
    chunks from each paper mention the concept. A concept is auto-resolved only when
    one paper clearly dominates; otherwise the caller must ask the user to clarify.
    """

    def __init__(
        self,
        graph: GraphDocument,
        *,
        dominance_ratio: float = 2.0,
        minimum_dominant_mentions: int = 2,
    ) -> None:
        self.graph = graph
        self.query = GraphQuery(graph)
        self.dominance_ratio = dominance_ratio
        self.minimum_dominant_mentions = minimum_dominant_mentions
        self.paper_titles = {
            str(node.properties.get("paper_id", node.node_id.removeprefix("paper:"))): (
                str(node.properties.get("title") or node.label)
            )
            for node in self.query.papers()
        }
        self.paper_nodes = {
            str(node.properties.get("paper_id", node.node_id.removeprefix("paper:"))): node
            for node in self.query.papers()
        }
        self.paper_aliases = self._build_paper_aliases()

    def resolve(self, query: str) -> EntityResolution:
        normalized_query = _normalize_text(query)
        if not normalized_query:
            return EntityResolution(status=EntityResolutionStatus.NO_MATCH)

        title_matches = self._match_paper_aliases(normalized_query)
        if len(title_matches) == 1:
            paper_id, aliases = next(iter(title_matches.items()))
            return EntityResolution(
                status=EntityResolutionStatus.RESOLVED,
                matched_entity=max(aliases, key=len),
                resolved_paper_id=paper_id,
                candidates=[self._candidate(paper_id, 100.0, aliases)],
                reason="query explicitly names a paper title or distinctive paper alias",
            )
        if len(title_matches) > 1:
            candidates = [
                self._candidate(paper_id, 100.0, aliases)
                for paper_id, aliases in title_matches.items()
            ]
            return self._ambiguous(candidates, "query contains aliases for multiple papers")

        concept_matches: list[tuple[str, Counter[str]]] = []
        for concept in self.graph.nodes:
            if concept.node_type != NodeType.CONCEPT:
                continue
            aliases = _concept_aliases(concept)
            matched_aliases = [alias for alias in aliases if _contains_alias(normalized_query, alias)]
            if not matched_aliases:
                continue
            matched = max(matched_aliases, key=len)
            mentions = self._concept_paper_mentions(concept.node_id)
            if mentions:
                concept_matches.append((matched, mentions))

        if not concept_matches:
            return EntityResolution(status=EntityResolutionStatus.NO_MATCH)

        # Prefer the concept occurring in the fewest papers. This prevents generic
        # terms such as "language model" from overpowering a named entity such as
        # "Q-Former" when both appear in the same question.
        minimum_document_frequency = min(len(mentions) for _, mentions in concept_matches)
        selected_matches = [
            item for item in concept_matches if len(item[1]) == minimum_document_frequency
        ]
        concept_scores: Counter[str] = Counter()
        entities_by_paper: dict[str, set[str]] = defaultdict(set)
        matched_concepts: list[str] = []
        for matched, mentions in selected_matches:
            matched_concepts.append(matched)
            for paper_id, count in mentions.items():
                concept_scores[paper_id] += count
                entities_by_paper[paper_id].add(matched)

        ranked = concept_scores.most_common()
        candidates = [
            self._candidate(paper_id, float(score), sorted(entities_by_paper[paper_id]))
            for paper_id, score in ranked
        ]
        top_paper, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        dominant = (
            len(ranked) == 1
            or (
                top_score >= self.minimum_dominant_mentions
                and top_score >= second_score * self.dominance_ratio
            )
        )
        if dominant:
            return EntityResolution(
                status=EntityResolutionStatus.RESOLVED,
                matched_entity=max(matched_concepts, key=len),
                resolved_paper_id=top_paper,
                candidates=candidates,
                reason="one paper has a clearly dominant concept-mention score",
            )
        return self._ambiguous(
            candidates,
            "the named concept is supported by multiple papers without a dominant match",
            matched_entity=max(matched_concepts, key=len),
        )

    def _build_paper_aliases(self) -> dict[str, set[str]]:
        aliases: dict[str, set[str]] = defaultdict(set)
        for paper_id, title in self.paper_titles.items():
            normalized_title = _normalize_text(title)
            aliases[paper_id].add(normalized_title)
            original_first_token = title.split(maxsplit=1)[0].strip(":")
            first_token = _normalize_text(original_first_token)
            if _looks_like_model_name(original_first_token):
                aliases[paper_id].add(first_token)
            source_path = str(self.paper_nodes[paper_id].properties.get("source_path", ""))
            source_stem = _normalize_text(Path(source_path).stem)
            if _is_distinctive_alias(source_stem):
                aliases[paper_id].add(source_stem)
        return aliases

    def _match_paper_aliases(self, query: str) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {}
        for paper_id, aliases in self.paper_aliases.items():
            found = [alias for alias in aliases if _contains_alias(query, alias)]
            if found:
                matches[paper_id] = found
        return matches

    def _concept_paper_mentions(self, concept_id: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for edge in self.query.in_edges.get(concept_id, []):
            if edge.edge_type != EdgeType.MENTIONS:
                continue
            source = self.query.nodes_by_id.get(edge.source_id)
            if source is None:
                continue
            if source.node_type == NodeType.CHUNK:
                paper_id = source.properties.get("paper_id")
                if paper_id:
                    counts[str(paper_id)] += 1
        return counts

    def _candidate(
        self,
        paper_id: str,
        score: float,
        matched_entities: list[str] | set[str],
    ) -> PaperEntityCandidate:
        return PaperEntityCandidate(
            paper_id=paper_id,
            title=self.paper_titles.get(paper_id, paper_id),
            score=score,
            matched_entities=sorted(set(matched_entities)),
        )

    @staticmethod
    def _ambiguous(
        candidates: list[PaperEntityCandidate],
        reason: str,
        *,
        matched_entity: str | None = None,
    ) -> EntityResolution:
        return EntityResolution(
            status=EntityResolutionStatus.AMBIGUOUS,
            matched_entity=matched_entity,
            candidates=sorted(candidates, key=lambda item: (-item.score, item.paper_id)),
            reason=reason,
        )


def load_graph_entity_resolver(
    *,
    graph_path: str | Path | None = None,
    workspace_path: str | Path | None = None,
) -> GraphEntityResolver | None:
    """Load the effective user graph first, falling back to the base graph."""

    if workspace_path:
        workspace = Path(workspace_path)
        if (workspace / "workspace.json").exists():
            return GraphEntityResolver(LocalGraphWorkspaceStore(workspace).load_effective_graph())
    if graph_path:
        graph = Path(graph_path)
        if (graph / "nodes.jsonl").exists() and (graph / "edges.jsonl").exists():
            return GraphEntityResolver(read_graph_jsonl(graph))
    return None


def _concept_aliases(node) -> set[str]:
    raw = {
        node.label,
        str(node.properties.get("canonical_name", "")),
        str(node.properties.get("normalized", "")).replace("-", " "),
        *[str(alias) for alias in node.properties.get("aliases", [])],
    }
    return {
        normalized
        for value in raw
        if (normalized := _normalize_text(value)) and _is_distinctive_alias(normalized)
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


def _contains_alias(query: str, alias: str) -> bool:
    if not alias:
        return False
    return re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", query) is not None


def _is_distinctive_alias(alias: str) -> bool:
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", alias)
    return len(compact) >= 3


def _looks_like_model_name(value: str) -> bool:
    uppercase_count = sum(character.isupper() for character in value)
    return _is_distinctive_alias(value) and (
        uppercase_count >= 2 or any(character.isdigit() for character in value) or "-" in value
    )

"""Deterministic evidence admission for model context windows."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from paper_agent.domain import EvidenceItem

from .contracts import ContextBudget, ScopeResolution


class ContextSelection(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    estimated_tokens: int = 0
    dropped_evidence_ids: list[str] = Field(default_factory=list)
    drop_reasons: dict[str, str] = Field(default_factory=dict)


class ContextManager:
    """Keep evidence compact, diverse and inside a per-node token allowance.

    The estimator is intentionally conservative and model-agnostic.  Exact model
    tokenizer accounting can be introduced behind this interface later.
    """

    def select_evidence(
        self,
        items: list[EvidenceItem],
        *,
        scope: ScopeResolution,
        budget: ContextBudget,
    ) -> ContextSelection:
        selected: list[EvidenceItem] = []
        dropped: list[str] = []
        reasons: dict[str, str] = {}
        seen_chunks: set[str] = set()
        total = 0

        for item in items:
            if not scope.allows(item.paper_id):
                dropped.append(item.evidence_id)
                reasons[item.evidence_id] = "outside_corpus_scope"
                continue
            if item.chunk_id in seen_chunks:
                dropped.append(item.evidence_id)
                reasons[item.evidence_id] = "duplicate_chunk"
                continue
            estimate = self.estimate_tokens(item.content)
            if estimate > budget.evidence_tokens - total:
                dropped.append(item.evidence_id)
                reasons[item.evidence_id] = "evidence_budget_exceeded"
                continue
            selected.append(item)
            seen_chunks.add(item.chunk_id)
            total += estimate

        return ContextSelection(
            items=selected,
            estimated_tokens=total,
            dropped_evidence_ids=dropped,
            drop_reasons=reasons,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate mixed Chinese/English token count without model loading."""

        if not text:
            return 0
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        non_cjk = len(text) - cjk
        words = len(re.findall(r"\S+", text))
        return max(words, cjk + (non_cjk + 3) // 4)

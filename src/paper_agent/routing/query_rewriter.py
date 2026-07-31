"""Conservative retrieval-query rewriting after evidence planning.

The first version is deliberately deterministic: it normalizes whitespace and
deduplicates planner-produced retrieval queries without changing named entities,
formula references or the user's final question.  An LLM rewriter can later be
implemented behind the same interface for low-confidence cases.
"""

from __future__ import annotations

import re

from paper_agent.routing.models import EvidencePlan, EvidenceSubQuestion


class QueryRewriter:
    def rewrite(self, plan: EvidencePlan) -> EvidencePlan:
        seen: set[str] = set()
        rewritten: list[EvidenceSubQuestion] = []
        for item in plan.sub_questions:
            query = re.sub(r"\s+", " ", item.query).strip()
            if not query or query.casefold() in seen:
                continue
            seen.add(query.casefold())
            rewritten.append(item.model_copy(update={"query": query}))
        # EvidencePlanner always produces at least one valid sub-question.  Keep
        # the original plan intact if a custom caller supplied only blank items.
        return plan.model_copy(update={"sub_questions": rewritten or plan.sub_questions})

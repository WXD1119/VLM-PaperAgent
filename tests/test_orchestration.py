from paper_agent.domain import ChunkKind, EvidenceItem
from paper_agent.orchestration import ContextBudget, ContextManager, CorpusScope, resolve_scope


def _evidence(evidence_id: str, paper_id: str, content: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        paper_id=paper_id,
        kind=ChunkKind.TEXT,
        pages=[1],
        content=content,
    )


def test_active_paper_scope_is_an_explicit_allow_list():
    scope = resolve_scope(CorpusScope.ACTIVE_PAPER_ONLY, active_paper_id="paper-current")
    selection = ContextManager().select_evidence(
        [_evidence("E1", "paper-current", "Allowed evidence"), _evidence("E2", "paper-library", "Forbidden evidence")],
        scope=scope,
        budget=ContextBudget(),
    )
    assert [item.evidence_id for item in selection.items] == ["E1"]
    assert selection.drop_reasons == {"E2": "outside_corpus_scope"}


def test_selected_scope_requires_explicit_papers():
    try:
        resolve_scope(CorpusScope.SELECTED_PAPERS)
    except ValueError as exc:
        assert "requires at least one" in str(exc)
    else:
        raise AssertionError("selected paper scope must require paper IDs")


def test_context_manager_drops_items_after_evidence_budget_is_full():
    budget = ContextBudget(
        total_tokens=2_000,
        system_tokens=100,
        state_tokens=100,
        memory_tokens=100,
        evidence_tokens=3,
        draft_tokens=100,
        output_reserve_tokens=256,
    )
    selection = ContextManager().select_evidence(
        [_evidence("E1", "paper-1", "one two"), _evidence("E2", "paper-1", "three four")],
        scope=resolve_scope(CorpusScope.LIBRARY_ONLY),
        budget=budget,
    )
    assert [item.evidence_id for item in selection.items] == ["E1"]
    assert selection.drop_reasons == {"E2": "evidence_budget_exceeded"}

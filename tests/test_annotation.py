import pytest

from paper_agent.evaluation.annotation import (
    append_case_unique,
    make_query_id,
    parse_selection,
)
from paper_agent.evaluation.golden import RetrievalCase
from scripts.annotate_retrieval import resolve_paper_id


def test_parse_selection_supports_lists_and_ranges() -> None:
    assert parse_selection("1,3-5", 5) == [1, 3, 4, 5]


def test_parse_selection_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_selection("1,4", 3)


def test_query_id_is_stable_and_duplicate_cases_are_rejected() -> None:
    query_id = make_query_id("What is Q-Former?", "paper_x")
    assert query_id == make_query_id("What is Q-Former?", "paper_x")
    cases: list[RetrievalCase] = []
    case = RetrievalCase(
        query_id=query_id,
        query="What is Q-Former?",
        paper_id="paper_x",
        relevant_chunk_ids={"chunk_x"},
    )
    append_case_unique(cases, case)
    with pytest.raises(ValueError, match="duplicate"):
        append_case_unique(cases, case)


def test_resolve_paper_id_from_paper_key_map() -> None:
    paper_id, label = resolve_paper_id(
        {"paper_key": "clip"},
        {"clip": {"paper_id": "paper_clip", "short_name": "CLIP"}},
    )

    assert paper_id == "paper_clip"
    assert label == "CLIP"


def test_resolve_paper_id_prefers_explicit_paper_id() -> None:
    paper_id, label = resolve_paper_id(
        {"paper_id": "paper_explicit", "paper_key": "clip"},
        {"clip": {"paper_id": "paper_clip", "short_name": "CLIP"}},
    )

    assert paper_id == "paper_explicit"
    assert label == "clip"

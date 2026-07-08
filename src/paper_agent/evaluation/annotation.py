import hashlib
import json
import re
from pathlib import Path

from pydantic import TypeAdapter

from paper_agent.evaluation.golden import RetrievalCase


def make_query_id(query: str, paper_id: str | None = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
    digest = hashlib.sha256(f"{paper_id or ''}\0{query}".encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'query'}_{digest}"


def parse_selection(value: str, candidate_count: int) -> list[int]:
    """Parse one-based comma/range selection, e.g. 1,3-5."""
    selected: set[int] = set()
    for part in value.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid descending range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    invalid = [index for index in selected if index < 1 or index > candidate_count]
    if invalid:
        raise ValueError(f"selection out of range: {invalid}")
    return sorted(selected)


def load_cases(path: Path) -> list[RetrievalCase]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[RetrievalCase]).validate_python(raw)


def save_cases_atomic(path: Path, cases: list[RetrievalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            [case.model_dump(mode="json") for case in cases],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_case_unique(cases: list[RetrievalCase], case: RetrievalCase) -> None:
    if any(existing.query_id == case.query_id for existing in cases):
        raise ValueError(f"duplicate query_id: {case.query_id}")
    if any(
        existing.query.strip().casefold() == case.query.strip().casefold()
        and existing.paper_id == case.paper_id
        for existing in cases
    ):
        raise ValueError("the same query and paper_id are already annotated")
    cases.append(case)

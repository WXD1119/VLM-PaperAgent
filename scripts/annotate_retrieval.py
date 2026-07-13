import argparse
import json
from pathlib import Path

from paper_agent.domain.chunk import ChunkKind
from paper_agent.evaluation.annotation import (
    append_case_unique,
    load_cases,
    make_query_id,
    parse_selection,
    save_cases_atomic,
)
from paper_agent.evaluation.golden import RetrievalCase
from paper_agent.retrieval import BM25Index, chunks_from_bundles, load_chunk_bundles


def load_question_queue(path: Path | None) -> list[dict]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("question queue must be a JSON array")
    return raw


def load_paper_id_map(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("paper id map must be a JSON array")
    mapping = {}
    for item in raw:
        paper_key = item.get("paper_key")
        paper_id = item.get("paper_id")
        if paper_key and paper_id:
            mapping[paper_key] = item
    return mapping


def resolve_paper_id(item: dict, paper_id_map: dict[str, dict]) -> tuple[str | None, str | None]:
    if item.get("paper_id"):
        return item["paper_id"], item.get("paper_short_name") or item.get("paper_key")
    paper_key = item.get("paper_key")
    if paper_key and paper_key in paper_id_map:
        row = paper_id_map[paper_key]
        return row.get("paper_id"), row.get("short_name") or paper_key
    return None, item.get("paper_short_name") or item.get("paper_key")


def show_hits(hits) -> None:
    for rank, hit in enumerate(hits, start=1):
        section = " > ".join(hit.section_path)
        content = " ".join(hit.content.split())
        context = " ".join(hit.context.split())
        print(f"\n[{rank}] score={hit.score:.4f} {hit.kind.value} pages={hit.pages}")
        print(f"paper={hit.paper_id}")
        print(f"section={section}")
        print(f"chunk_id={hit.chunk_id}")
        print(f"content={content[:900]}")
        if context:
            print(f"context={context[:500]}")


def annotate_one(index: BM25Index, item: dict, top_k: int, annotator: str, paper_id_map: dict[str, dict] | None = None):
    paper_id_map = paper_id_map or {}
    query = item.get("query") or input("\nQuery (q=quit): ").strip()
    if query.lower() == "q":
        return "quit", None
    if not query:
        return "skip", None

    resolved_paper_id, paper_label = resolve_paper_id(item, paper_id_map)
    paper_id = resolved_paper_id
    if paper_id:
        label = f" [{paper_label}]" if paper_label else ""
        print(f"Resolved paper_id{label}: {paper_id}")
    else:
        paper_id = input("paper_id (optional): ").strip() or None
    kind_value = item.get("kind") or input(
        "kind [text/equation/table/figure] (optional): "
    ).strip()
    kind = ChunkKind(kind_value) if kind_value else None
    category = item.get("category", "unspecified")

    hits = index.search(query, top_k=top_k, paper_id=paper_id, kind=kind)
    if not hits:
        print("No candidates. Enter s to skip or q to quit.")
        action = input("> ").strip().lower()
        return ("quit" if action == "q" else "skip"), None
    print(f"\nQuery: {query}")
    paper_display = paper_id or "*"
    if paper_label:
        paper_display = f"{paper_display} ({paper_label})"
    print(f"Category: {category} | paper={paper_display} | kind={kind_value or '*'}")
    show_hits(hits)

    while True:
        selection = input(
            "\nRelevant ranks (e.g. 1,3-4), s=skip, q=save & quit: "
        ).strip().lower()
        if selection == "s":
            return "skip", None
        if selection == "q":
            return "quit", None
        try:
            ranks = parse_selection(selection, len(hits))
            if not ranks:
                raise ValueError("select at least one relevant candidate")
            break
        except (ValueError, TypeError) as exc:
            print(f"Invalid selection: {exc}")

    expected_answer = input("Expected answer summary (recommended): ").strip()
    notes = input("Annotation notes (optional): ").strip()
    case = RetrievalCase(
        query_id=item.get("query_id") or make_query_id(query, paper_id),
        query=query,
        relevant_chunk_ids={hits[rank - 1].chunk_id for rank in ranks},
        paper_id=paper_id,
        kind=kind,
        category=category,
        annotator=annotator,
        expected_answer=expected_answer,
        notes=notes,
    )
    return "saved", case


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively annotate retrieval Golden Set")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--paper-id-map", type=Path)
    parser.add_argument("--annotator", default="human")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cases = load_cases(args.output)
    if not cases and args.seed:
        cases = load_cases(args.seed)
        save_cases_atomic(args.output, cases)
    completed_ids = {case.query_id for case in cases}
    paper_id_map = load_paper_id_map(args.paper_id_map)
    queue = [
        item
        for item in load_question_queue(args.questions)
        if item.get("query_id") not in completed_ids
    ]
    if not queue:
        queue = [{}]

    index = BM25Index(chunks_from_bundles(load_chunk_bundles(args.chunks)))
    print(f"Loaded {len(index.chunks)} chunks and {len(cases)} existing annotations.")
    if paper_id_map:
        print(f"Loaded {len(paper_id_map)} paper_id mappings.")
    for item in queue:
        action, case = annotate_one(index, item, args.top_k, args.annotator, paper_id_map)
        if action == "quit":
            break
        if action == "saved" and case is not None:
            append_case_unique(cases, case)
            save_cases_atomic(args.output, cases)
            print(f"Saved {case.query_id}. Total annotations: {len(cases)}")
    print(f"Golden Set: {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

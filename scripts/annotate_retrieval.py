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


def annotate_one(index: BM25Index, item: dict, top_k: int, annotator: str):
    query = item.get("query") or input("\nQuery (q=quit): ").strip()
    if query.lower() == "q":
        return "quit", None
    if not query:
        return "skip", None

    paper_id = item.get("paper_id") or input("paper_id (optional): ").strip() or None
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
    print(f"Category: {category} | paper={paper_id or '*'} | kind={kind_value or '*'}")
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
    parser.add_argument("--annotator", default="human")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cases = load_cases(args.output)
    if not cases and args.seed:
        cases = load_cases(args.seed)
        save_cases_atomic(args.output, cases)
    completed_ids = {case.query_id for case in cases}
    queue = [
        item
        for item in load_question_queue(args.questions)
        if item.get("query_id") not in completed_ids
    ]
    if not queue:
        queue = [{}]

    index = BM25Index(chunks_from_bundles(load_chunk_bundles(args.chunks)))
    print(f"Loaded {len(index.chunks)} chunks and {len(cases)} existing annotations.")
    for item in queue:
        action, case = annotate_one(index, item, args.top_k, args.annotator)
        if action == "quit":
            break
        if action == "saved" and case is not None:
            append_case_unique(cases, case)
            save_cases_atomic(args.output, cases)
            print(f"Saved {case.query_id}. Total annotations: {len(cases)}")
    print(f"Golden Set: {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

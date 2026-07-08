import argparse
import json
from pathlib import Path

from paper_agent.domain.paper import Paper
from paper_agent.ingestion.chunker import PaperChunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Create retrieval chunks from paper.json")
    parser.add_argument("--paper", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-chars", type=int, default=2800)
    parser.add_argument("--overlap-elements", type=int, default=1)
    args = parser.parse_args()

    paper = Paper.model_validate_json(args.paper.read_text(encoding="utf-8"))
    bundle = PaperChunker(
        max_chars=args.max_chars,
        overlap_elements=args.overlap_elements,
    ).chunk(paper)
    output = args.output or args.paper.with_name("chunks.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kinds: dict[str, int] = {}
    for chunk in bundle.children:
        kinds[chunk.kind.value] = kinds.get(chunk.kind.value, 0) + 1
    print(f"paper_id: {bundle.paper_id}")
    print(f"parents: {len(bundle.parents)}")
    print(f"children: {len(bundle.children)}")
    print(f"kinds: {kinds}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()

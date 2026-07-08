import argparse

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval import BM25Index, chunks_from_bundles, load_chunk_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Search paper chunks with BM25")
    parser.add_argument("--chunks", required=True, help="chunks.json file or parent directory")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper-id")
    parser.add_argument("--kind", choices=[kind.value for kind in ChunkKind])
    args = parser.parse_args()

    bundles = load_chunk_bundles(args.chunks)
    index = BM25Index(chunks_from_bundles(bundles))
    kind = ChunkKind(args.kind) if args.kind else None
    hits = index.search(
        args.query,
        top_k=args.top_k,
        paper_id=args.paper_id,
        kind=kind,
    )

    if not hits:
        print("No lexical matches found.")
        return
    for rank, hit in enumerate(hits, start=1):
        section = " > ".join(hit.section_path)
        snippet = " ".join(hit.content.split())[:500]
        print(f"\n[{rank}] score={hit.score:.4f} kind={hit.kind.value}")
        print(f"paper_id={hit.paper_id} pages={hit.pages}")
        print(f"section={section}")
        print(f"chunk_id={hit.chunk_id}")
        print(f"content={snippet}")


if __name__ == "__main__":
    main()

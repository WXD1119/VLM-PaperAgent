import argparse
from pathlib import Path

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval import SentenceTransformerEncoder
from paper_agent.storage import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Chroma with dense embeddings")
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper-id")
    parser.add_argument("--kind", choices=[kind.value for kind in ChunkKind])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    encoder = SentenceTransformerEncoder(
        args.model,
        device=args.device,
        local_files_only=args.offline,
    )
    store = ChromaVectorStore(args.db, args.collection, encoder)
    kind = ChunkKind(args.kind) if args.kind else None
    hits = store.search(args.query, args.top_k, args.paper_id, kind)
    for rank, hit in enumerate(hits, start=1):
        print(f"\n[{rank}] cosine={hit.score:.4f} kind={hit.kind.value}")
        print(f"paper_id={hit.paper_id} pages={hit.pages}")
        print(f"section={' > '.join(hit.section_path)}")
        print(f"chunk_id={hit.chunk_id}")
        print(f"content={' '.join(hit.content.split())[:500]}")


if __name__ == "__main__":
    main()

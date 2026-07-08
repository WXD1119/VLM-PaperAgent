import argparse
from pathlib import Path

from paper_agent.domain.chunk import ChunkKind
from paper_agent.retrieval import (
    BM25Index,
    HybridRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search with BM25 + Dense RRF")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--paper-id")
    parser.add_argument("--kind", choices=[kind.value for kind in ChunkKind])
    args = parser.parse_args()

    chunks = chunks_from_bundles(load_chunk_bundles(args.chunks))
    sparse = BM25Index(chunks)
    encoder = SentenceTransformerEncoder(
        args.model,
        device=args.device,
        local_files_only=args.offline,
    )
    dense = ChromaVectorStore(args.db, args.collection, encoder)
    hybrid = HybridRetriever(sparse, dense, args.rrf_k, args.candidate_k)
    kind = ChunkKind(args.kind) if args.kind else None
    hits = hybrid.search(args.query, args.top_k, args.paper_id, kind)

    for rank, hit in enumerate(hits, start=1):
        print(
            f"\n[{rank}] rrf={hit.score:.6f} "
            f"sparse_rank={hit.sparse_rank} dense_rank={hit.dense_rank}"
        )
        print(f"paper_id={hit.paper_id} kind={hit.kind.value} pages={hit.pages}")
        print(f"section={' > '.join(hit.section_path)}")
        print(f"chunk_id={hit.chunk_id}")
        print(f"content={' '.join(hit.content.split())[:500]}")


if __name__ == "__main__":
    main()

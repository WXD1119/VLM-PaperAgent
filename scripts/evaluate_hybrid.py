import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from paper_agent.evaluation import RetrievalCase, evaluate_retriever
from paper_agent.retrieval import (
    BM25Index,
    HybridRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 + Dense RRF")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = TypeAdapter(list[RetrievalCase]).validate_python(
        json.loads(args.golden.read_text(encoding="utf-8"))
    )
    chunks = chunks_from_bundles(load_chunk_bundles(args.chunks))
    sparse = BM25Index(chunks)
    encoder = SentenceTransformerEncoder(
        args.model,
        device=args.device,
        local_files_only=args.offline,
    )
    dense = ChromaVectorStore(args.db, args.collection, encoder)
    collection_count = dense.count()
    print(f"chunks: {len(chunks)}")
    print(f"collection_count: {collection_count}")
    if collection_count == 0:
        raise SystemExit(
            "Chroma collection is empty; run scripts/index_dense.py before hybrid evaluation."
        )
    if collection_count != len(chunks):
        print(
            "WARNING: Chroma collection count differs from loaded chunks; "
            "dense retrieval may be stale. Re-run scripts/index_dense.py if this is unexpected."
        )
    hybrid = HybridRetriever(sparse, dense, args.rrf_k, args.candidate_k)
    result = evaluate_retriever(hybrid, cases, top_k=args.top_k)

    print(f"cases: {len(result.cases)}")
    print(f"Hit@1: {result.mean_hit_at_1:.4f}")
    print(f"Hit@5: {result.mean_hit_at_5:.4f}")
    print(f"Recall@1: {result.macro_recall_at_1:.4f}")
    print(f"Recall@5: {result.macro_recall_at_5:.4f}")
    print(f"MRR: {result.mean_reciprocal_rank:.4f}")
    print(f"nDCG@5: {result.mean_ndcg_at_5:.4f}")
    for case in result.cases:
        print(
            f"{case.query_id}: H@1={case.hit_at_1:.0f} H@5={case.hit_at_5:.0f} "
            f"R@1={case.recall_at_1:.3f} "
            f"R@5={case.recall_at_5:.3f} RR={case.reciprocal_rank:.3f} "
            f"nDCG@5={case.ndcg_at_5:.3f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"output: {args.output}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from paper_agent.evaluation import RetrievalCase, evaluate_retriever
from paper_agent.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval + cross-encoder reranking")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
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
        args.embedding_model, device=args.device, local_files_only=args.offline
    )
    dense = ChromaVectorStore(args.db, args.collection, encoder)
    hybrid = HybridRetriever(sparse, dense, args.rrf_k, args.candidate_k)
    cross_encoder = CrossEncoderReranker(
        args.reranker_model, device=args.device, local_files_only=args.offline
    )
    retriever = RerankedRetriever(hybrid, cross_encoder, args.candidate_k)
    result = evaluate_retriever(retriever, cases, top_k=args.top_k)

    print(f"cases: {len(result.cases)}")
    print(f"Recall@1: {result.macro_recall_at_1:.4f}")
    print(f"Recall@5: {result.macro_recall_at_5:.4f}")
    print(f"MRR: {result.mean_reciprocal_rank:.4f}")
    print(f"nDCG@5: {result.mean_ndcg_at_5:.4f}")
    for case in result.cases:
        print(
            f"{case.query_id}: R@1={case.recall_at_1:.3f} "
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

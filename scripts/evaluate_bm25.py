import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from paper_agent.evaluation import RetrievalCase, evaluate_bm25
from paper_agent.retrieval import BM25Index, chunks_from_bundles, load_chunk_bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 against a retrieval Golden Set")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    raw_cases = json.loads(args.golden.read_text(encoding="utf-8"))
    cases = TypeAdapter(list[RetrievalCase]).validate_python(raw_cases)
    index = BM25Index(chunks_from_bundles(load_chunk_bundles(args.chunks)))
    result = evaluate_bm25(index, cases, top_k=args.top_k)

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

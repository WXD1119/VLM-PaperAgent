import argparse

from paper_agent.memory import SummaryMemoryStore, build_summary_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recall relevant episodic summaries")
    parser.add_argument("--query", required=True)
    parser.add_argument("--summaries", default="artifacts/memory/summaries.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = SummaryMemoryStore(args.summaries).list()
    store = build_summary_vector_store(summaries)
    hits = store.search(args.query, top_k=args.top_k)

    print(f"summaries: {len(summaries)}")
    print(f"hits: {len(hits)}")
    for index, hit in enumerate(hits, start=1):
        print(f"\n[{index}] score={hit.score:.4f} summary_id={hit.summary.summary_id}")
        print(f"paper_ids={hit.summary.paper_ids}")
        print(f"key_entities={hit.summary.key_entities}")
        print(hit.summary.summary)


if __name__ == "__main__":
    main()

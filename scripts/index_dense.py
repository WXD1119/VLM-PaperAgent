import argparse
from pathlib import Path

from paper_agent.retrieval import (
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed paper chunks into Chroma")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the target Chroma collection before indexing.",
    )
    args = parser.parse_args()

    chunks = chunks_from_bundles(load_chunk_bundles(args.chunks))
    encoder = SentenceTransformerEncoder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.offline,
    )
    store = ChromaVectorStore(args.db, args.collection, encoder, reset=args.reset)
    indexed = store.upsert(chunks, batch_size=args.batch_size)
    print(f"model: {encoder.model_name}")
    print(f"dimension: {encoder.dimension}")
    print(f"indexed: {indexed}")
    print(f"collection_count: {store.collection.count()}")
    print(f"database: {args.db}")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON array")
    return raw


def paper_id_from_pdf(pdf: Path) -> str | None:
    if not pdf.exists():
        return None
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"paper_{digest.hexdigest()[:16]}"


def benchmark_paper_ids(manifest: Path, pdf_dir: Path) -> list[str]:
    ids: list[str] = []
    for paper in load_manifest(manifest):
        paper_id = paper_id_from_pdf(pdf_dir / paper["filename"])
        if paper_id is not None:
            ids.append(paper_id)
    return ids


def inspect_chunks(paper_ids: list[str], papers_root: Path) -> dict[str, bool]:
    return {
        paper_id: (papers_root / paper_id / "chunks.json").exists()
        for paper_id in paper_ids
    }


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess parsed VLM benchmark papers: index, graph and commands"
    )
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw/vlm_benchmark"))
    parser.add_argument("--papers", type=Path, default=Path("artifacts/papers"))
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma_vlm_benchmark"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3_vlm_v2")
    parser.add_argument("--embedding-model", default="artifacts/models/bge-m3")
    parser.add_argument("--reranker-model", default="artifacts/models/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--graph-output", type=Path, default=Path("artifacts/graph_vlm_benchmark"))
    parser.add_argument("--questions", type=Path, default=Path("evals/retrieval_questions.v2.draft.json"))
    parser.add_argument("--paper-id-map", type=Path, default=Path("evals/vlm_paper_id_map.v2.json"))
    parser.add_argument("--golden", type=Path, default=Path("evals/retrieval_golden.v2.json"))
    parser.add_argument("--eval-output-dir", type=Path, default=Path("artifacts/evals/vlm_v2"))
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_ids = benchmark_paper_ids(args.manifest, args.pdf_dir)
    chunk_status = inspect_chunks(paper_ids, args.papers)
    ready = [paper_id for paper_id, exists in chunk_status.items() if exists]
    missing = [paper_id for paper_id, exists in chunk_status.items() if not exists]
    print(f"benchmark_papers: {len(paper_ids)}")
    print(f"chunks_ready: {len(ready)}")
    print(f"chunks_missing: {len(missing)}")
    for paper_id in missing:
        print(f"- missing chunks: {paper_id}")
    dry_run = not args.execute
    if dry_run:
        print("mode: dry-run; pass --execute to run index/graph commands")
    if missing:
        print("warning: some benchmark papers are not chunked yet; index/graph will include available chunks only")

    if not args.skip_index:
        run(
            [
                sys.executable,
                "scripts/index_dense.py",
                "--chunks",
                str(args.papers),
                "--db",
                str(args.db),
                "--collection",
                args.collection,
                "--model",
                args.embedding_model,
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--offline",
            ],
            dry_run=dry_run,
        )
    if not args.skip_graph:
        run(
            [
                sys.executable,
                "scripts/build_graph.py",
                "--papers",
                str(args.papers),
                "--chunks",
                str(args.papers),
                "--output",
                str(args.graph_output),
                "--allow-empty-answers",
            ],
            dry_run=dry_run,
        )

    print("\nBuild paper_id map:")
    print(
        " ".join(
            [
                sys.executable,
                "scripts/build_vlm_paper_id_map.py",
                "--manifest",
                str(args.manifest),
                "--papers",
                str(args.papers),
                "--output",
                str(args.paper_id_map),
            ]
        )
    )
    print("\nAnnotation command:")
    print(
        " ".join(
            [
                sys.executable,
                "scripts/annotate_retrieval.py",
                "--chunks",
                str(args.papers),
                "--questions",
                str(args.questions),
                "--paper-id-map",
                str(args.paper_id_map),
                "--output",
                str(args.golden),
                "--annotator",
                "wxd",
                "--top-k",
                str(args.top_k),
            ]
        )
    )
    print("\nAfter annotation, evaluate:")
    print(
        " ".join(
            [
                sys.executable,
                "scripts/evaluate_reranked.py",
                "--chunks",
                str(args.papers),
                "--db",
                str(args.db),
                "--collection",
                args.collection,
                "--embedding-model",
                args.embedding_model,
                "--reranker-model",
                args.reranker_model,
                "--offline",
                "--device",
                args.device,
                "--golden",
                str(args.golden),
                "--top-k",
                "5",
                "--candidate-k",
                "20",
                "--output",
                str(args.eval_output_dir / "reranked.v2.json"),
            ]
        )
    )


if __name__ == "__main__":
    main()

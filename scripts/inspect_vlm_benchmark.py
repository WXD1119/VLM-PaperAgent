import argparse
import hashlib
import json
from pathlib import Path


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


def find_content_list(mineru_output: Path, pdf_stem: str) -> Path | None:
    matches = list(mineru_output.rglob(f"{pdf_stem}_content_list.json"))
    return matches[0] if len(matches) == 1 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect VLM benchmark preparation status")
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw/vlm_benchmark"))
    parser.add_argument("--mineru-output", type=Path, default=Path("artifacts/mineru_vlm_benchmark"))
    parser.add_argument("--papers-output", type=Path, default=Path("artifacts/papers"))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in load_manifest(args.manifest):
        pdf = args.pdf_dir / item["filename"]
        paper_id = paper_id_from_pdf(pdf)
        content_list = find_content_list(args.mineru_output, pdf.stem)
        paper_json = args.papers_output / paper_id / "paper.json" if paper_id else None
        chunks_json = args.papers_output / paper_id / "chunks.json" if paper_id else None
        rows.append(
            {
                "paper_key": item["paper_key"],
                "short_name": item["short_name"],
                "pdf": str(pdf),
                "pdf_exists": pdf.exists(),
                "paper_id": paper_id,
                "mineru_content_list": str(content_list) if content_list else None,
                "mineru_ready": content_list is not None,
                "paper_json": str(paper_json) if paper_json else None,
                "paper_ready": bool(paper_json and paper_json.exists()),
                "chunks_json": str(chunks_json) if chunks_json else None,
                "chunks_ready": bool(chunks_json and chunks_json.exists()),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"papers: {len(rows)}")
    print(f"pdf_ready: {sum(row['pdf_exists'] for row in rows)}")
    print(f"mineru_ready: {sum(row['mineru_ready'] for row in rows)}")
    print(f"paper_ready: {sum(row['paper_ready'] for row in rows)}")
    print(f"chunks_ready: {sum(row['chunks_ready'] for row in rows)}")
    for row in rows:
        status = []
        status.append("pdf" if row["pdf_exists"] else "no_pdf")
        status.append("mineru" if row["mineru_ready"] else "no_mineru")
        status.append("paper" if row["paper_ready"] else "no_paper")
        status.append("chunks" if row["chunks_ready"] else "no_chunks")
        print(f"{row['short_name']}: {', '.join(status)}")


if __name__ == "__main__":
    main()

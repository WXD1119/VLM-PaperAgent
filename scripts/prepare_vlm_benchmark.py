import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_vlm_benchmark_questions import build_questions, load_manifest


@dataclass(frozen=True)
class BenchmarkPaper:
    paper_key: str
    short_name: str
    title: str
    filename: str
    pdf_path: Path


def papers_from_manifest(manifest: Path, pdf_dir: Path) -> list[BenchmarkPaper]:
    papers: list[BenchmarkPaper] = []
    for item in load_manifest(manifest):
        papers.append(
            BenchmarkPaper(
                paper_key=item["paper_key"],
                short_name=item["short_name"],
                title=item["title"],
                filename=item["filename"],
                pdf_path=pdf_dir / item["filename"],
            )
        )
    return papers


def find_content_list(mineru_output: Path, pdf_stem: str) -> Path | None:
    matches = list(mineru_output.rglob(f"{pdf_stem}_content_list.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def write_question_draft(manifest: Path, output: Path, *, dry_run: bool) -> None:
    questions = build_questions(load_manifest(manifest))
    print(f"question_draft: {output} ({len(questions)} questions)")
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_and_chunk(
    papers: list[BenchmarkPaper],
    *,
    mineru_output: Path,
    papers_output: Path,
    mineru_version: str,
    max_chars: int,
    dry_run: bool,
) -> tuple[int, int]:
    parsed = chunked = 0
    for paper in papers:
        content_list = find_content_list(mineru_output, paper.pdf_path.stem)
        if content_list is None:
            print(f"missing_mineru_output: {paper.short_name} expected *{paper.pdf_path.stem}_content_list.json")
            continue
        paper_json = papers_output / paper_id_from_pdf(paper.pdf_path) / "paper.json"
        run_command(
            [
                sys.executable,
                "scripts/parse_paper.py",
                "--pdf",
                str(paper.pdf_path),
                "--mineru-output",
                str(mineru_output),
                "--output",
                str(papers_output),
                "--mineru-version",
                mineru_version,
            ],
            dry_run=dry_run,
        )
        parsed += 1
        if dry_run:
            continue
        if not paper_json.exists():
            raise FileNotFoundError(f"parse did not produce expected file: {paper_json}")
        run_command(
            [
                sys.executable,
                "scripts/chunk_paper.py",
                "--paper",
                str(paper_json),
                "--max-chars",
                str(max_chars),
            ],
            dry_run=False,
        )
        chunked += 1
    return parsed, chunked


def paper_id_from_pdf(pdf: Path) -> str:
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"paper_{digest.hexdigest()[:16]}"


def print_mineru_command(pdf_dir: Path, mineru_output: Path, *, backend: str) -> None:
    print("MinerU command:")
    print(
        "CUDA_VISIBLE_DEVICES=2 "
        "MINERU_MODEL_SOURCE=modelscope "
        f"mineru -p {pdf_dir} -o {mineru_output} -b {backend} -m auto"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 10-paper VLM benchmark")
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw/vlm_benchmark"))
    parser.add_argument("--mineru-output", type=Path, default=Path("artifacts/mineru_vlm_benchmark"))
    parser.add_argument("--papers-output", type=Path, default=Path("artifacts/papers"))
    parser.add_argument("--questions-output", type=Path, default=Path("evals/retrieval_questions.v2.draft.json"))
    parser.add_argument("--mineru-version", default="3.4.2")
    parser.add_argument("--mineru-backend", default="pipeline")
    parser.add_argument("--max-chars", type=int, default=2800)
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument("--skip-parse-chunk", action="store_true")
    parser.add_argument("--print-mineru-command", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers = papers_from_manifest(args.manifest, args.pdf_dir)
    missing_pdfs = [paper.pdf_path for paper in papers if not paper.pdf_path.exists()]
    print(f"papers: {len(papers)}")
    print(f"missing_pdfs: {len(missing_pdfs)}")
    for path in missing_pdfs:
        print(f"- {path}")
    if args.print_mineru_command:
        print_mineru_command(args.pdf_dir, args.mineru_output, backend=args.mineru_backend)
    if not args.skip_questions:
        write_question_draft(args.manifest, args.questions_output, dry_run=args.dry_run)
    if not args.skip_parse_chunk:
        parsed, chunked = parse_and_chunk(
            papers,
            mineru_output=args.mineru_output,
            papers_output=args.papers_output,
            mineru_version=args.mineru_version,
            max_chars=args.max_chars,
            dry_run=args.dry_run,
        )
        print(f"parse_jobs: {parsed}")
        print(f"chunk_jobs: {chunked}")


if __name__ == "__main__":
    main()

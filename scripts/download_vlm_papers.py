import argparse
import json
import urllib.request
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON array")
    return raw


def download_file(url: str, output: Path, *, timeout: float = 120) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VLM-PaperAgent benchmark downloader"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        output.write_bytes(response.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VLM benchmark PDFs from manifest")
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/vlm_benchmark"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = skipped = failed = 0
    print(f"papers: {len(papers)}")
    print(f"output_dir: {args.output_dir}")
    for index, paper in enumerate(papers, start=1):
        output = args.output_dir / paper["filename"]
        url = paper["pdf_url"]
        label = f"{paper['short_name']} ({paper['arxiv_id']})"
        if output.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(papers)}] skip existing: {label} -> {output}")
            continue
        print(f"[{index}/{len(papers)}] download: {label}")
        print(f"  url: {url}")
        print(f"  out: {output}")
        if args.dry_run:
            continue
        try:
            download_file(url, output)
            completed += 1
            print(f"  written: {output} ({output.stat().st_size} bytes)")
        except Exception as exc:
            failed += 1
            print(f"  failed: {exc}")
    print(f"completed: {completed}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

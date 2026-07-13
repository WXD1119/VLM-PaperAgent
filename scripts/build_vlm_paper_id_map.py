import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def normalize_title(value: str) -> str:
    return " ".join(value.lower().replace(":", " ").split())


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON array")
    return raw


def load_papers(root: Path) -> list[dict]:
    papers = []
    for path in sorted(root.rglob("paper.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_paper_json_path"] = str(path)
        papers.append(data)
    return papers


def match_paper(manifest_item: dict, papers: list[dict]) -> dict | None:
    target = normalize_title(manifest_item["title"])
    short = normalize_title(manifest_item["short_name"])
    for paper in papers:
        title = normalize_title(paper.get("title", ""))
        if title == target or target in title or title in target:
            return paper
    for paper in papers:
        title = normalize_title(paper.get("title", ""))
        if short and short in title:
            return paper
    return None


def build_map(manifest: Path, papers_root: Path) -> list[dict]:
    papers = load_papers(papers_root)
    rows: list[dict] = []
    for item in load_manifest(manifest):
        paper = match_paper(item, papers)
        rows.append(
            {
                "paper_key": item["paper_key"],
                "short_name": item["short_name"],
                "title": item["title"],
                "arxiv_id": item["arxiv_id"],
                "paper_id": paper.get("paper_id") if paper else None,
                "matched_title": paper.get("title") if paper else None,
                "source_path": paper.get("source_path") if paper else None,
                "paper_json_path": paper.get("_paper_json_path") if paper else None,
                "status": "matched" if paper else "missing",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM benchmark paper_key -> paper_id map")
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--papers", type=Path, default=Path("artifacts/papers"))
    parser.add_argument("--output", type=Path, default=Path("evals/vlm_paper_id_map.v2.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_map(args.manifest, args.papers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"papers: {len(rows)}")
    print(f"matched: {sum(row['status'] == 'matched' for row in rows)}")
    print(f"missing: {sum(row['status'] == 'missing' for row in rows)}")
    for row in rows:
        print(f"{row['short_name']}: {row['status']} {row['paper_id'] or ''}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

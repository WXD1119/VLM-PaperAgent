import argparse
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest must be a JSON array")
    return raw


def build_questions(papers: list[dict]) -> list[dict]:
    questions: list[dict] = []
    for paper in papers:
        for seed in paper.get("question_seeds", []):
            questions.append(
                {
                    "query_id": seed["query_id"],
                    "query": seed["query"],
                    "paper_key": paper["paper_key"],
                    "paper_title": paper["title"],
                    "paper_short_name": paper["short_name"],
                    "arxiv_id": paper["arxiv_id"],
                    "category": seed.get("category", "unspecified"),
                    "kind": seed.get("kind"),
                    "expected_answer": seed.get("expected_answer", ""),
                    "source": "vlm_benchmark_manifest.v2",
                    "annotation_status": "draft",
                    "notes": "Candidate question; relevant_chunk_ids must be verified by human annotation.",
                }
            )
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate VLM benchmark questions")
    parser.add_argument("--manifest", type=Path, default=Path("evals/vlm_benchmark_manifest.v2.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/retrieval_questions.v2.draft.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = build_questions(load_manifest(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"papers: {len(load_manifest(args.manifest))}")
    print(f"questions: {len(questions)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

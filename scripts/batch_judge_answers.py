import argparse
from dataclasses import dataclass
from pathlib import Path

from paper_agent.agents import SemanticCitationJudge
from paper_agent.domain import AnswerBundle, SemanticCitationReport
from paper_agent.llm import RemoteStructuredClient


@dataclass(frozen=True)
class JudgeJob:
    case_id: str
    answer_path: Path
    output_path: Path


def case_id_from_answer(path: Path) -> str:
    return path.name.removesuffix(".answer.json")


def select_judge_jobs(
    answer_paths: list[Path],
    *,
    output_dir: Path,
    overwrite: bool = False,
    include_abstained: bool = False,
    limit: int | None = None,
) -> list[JudgeJob]:
    jobs: list[JudgeJob] = []
    for path in sorted(answer_paths):
        case_id = case_id_from_answer(path)
        output_path = output_dir / f"{case_id}.glm-judge.json"
        if output_path.exists() and not overwrite:
            continue
        bundle = AnswerBundle.model_validate_json(path.read_text(encoding="utf-8"))
        if bundle.answer.abstained and not include_abstained:
            continue
        if not bundle.answer.claims and not include_abstained:
            continue
        jobs.append(JudgeJob(case_id=case_id, answer_path=path, output_path=output_path))
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch semantic citation judging for answers")
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-abstained",
        action="store_true",
        help="Also write empty judge reports for abstained answers.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after an individual answer fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer_paths = sorted(args.answers.glob("*.answer.json"))
    jobs = select_judge_jobs(
        answer_paths,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        include_abstained=args.include_abstained,
        limit=args.limit,
    )
    print(f"answers: {len(answer_paths)}")
    print(f"jobs: {len(jobs)}")
    for job in jobs:
        print(f"- {job.case_id}: {job.answer_path} -> {job.output_path}")
    if args.dry_run:
        print("dry_run: true")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = RemoteStructuredClient(args.judge_url, timeout=args.timeout)
    print(f"judge_endpoint: {client.url}")
    judge = SemanticCitationJudge(client)
    failures: list[tuple[str, str]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] judging {job.case_id}")
        try:
            bundle = AnswerBundle.model_validate_json(job.answer_path.read_text(encoding="utf-8"))
            if bundle.answer.abstained:
                report = SemanticCitationReport()
            else:
                report = judge.evaluate(bundle.answer, bundle.evidence_pack)
            job.output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            print(f"written: {job.output_path}")
        except Exception as exc:
            failures.append((job.case_id, str(exc)))
            print(f"failed: {job.case_id}: {exc}")
            if not args.keep_going:
                raise

    print(f"\ncompleted: {len(jobs) - len(failures)}")
    print(f"failed: {len(failures)}")
    if failures:
        for case_id, error in failures:
            print(f"- {case_id}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

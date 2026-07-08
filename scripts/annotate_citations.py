import argparse
from pathlib import Path

from paper_agent.domain import AnswerBundle, SupportVerdict
from paper_agent.evaluation.citation import (
    CitationCaseLabel,
    CitationClaimLabel,
    CitationGoldenSet,
)


CHOICES = {
    "s": SupportVerdict.SUPPORTED,
    "p": SupportVerdict.PARTIALLY_SUPPORTED,
    "u": SupportVerdict.UNSUPPORTED,
}


def save_atomic(path: Path, golden: CitationGoldenSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(golden.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def load_golden(path: Path, annotator: str) -> CitationGoldenSet:
    if not path.exists():
        return CitationGoldenSet(annotator=annotator)
    golden = CitationGoldenSet.model_validate_json(path.read_text(encoding="utf-8"))
    if golden.annotator != annotator:
        raise ValueError(
            f"golden set annotator is {golden.annotator!r}, not requested {annotator!r}"
        )
    return golden


def prompt_verdict() -> SupportVerdict:
    while True:
        value = input("Verdict [s=supported, p=partial, u=unsupported]: ").strip().lower()
        if value in CHOICES:
            return CHOICES[value]
        print("Please enter s, p, or u.")


def annotate_bundle(case_id: str, bundle: AnswerBundle) -> CitationCaseLabel:
    pack = bundle.evidence_pack
    print("\n" + "=" * 80)
    print(f"Case: {case_id}")
    print(f"Query: {pack.query}")
    print(f"Answer: {bundle.answer.answer}")
    print(f"Abstained: {bundle.answer.abstained}")
    while True:
        value = input("Was the abstention decision correct? [y/n]: ").strip().lower()
        if value in {"y", "n"}:
            abstention_correct = value == "y"
            break
    evidence = {item.evidence_id: item for item in pack.items}
    labels: list[CitationClaimLabel] = []
    for index, claim in enumerate(bundle.answer.claims, start=1):
        print(f"\nClaim {index}: {claim.text}")
        for evidence_id in claim.evidence_ids:
            item = evidence[evidence_id]
            print(
                f"[{evidence_id}] paper={item.paper_id} pages={item.pages} "
                f"section={' > '.join(item.section_path)}"
            )
            print(item.content)
        labels.append(
            CitationClaimLabel(
                claim_index=index,
                verdict=prompt_verdict(),
                notes=input("Notes (optional): ").strip(),
            )
        )
    return CitationCaseLabel(
        case_id=case_id,
        query=pack.query,
        answer_abstained=bundle.answer.abstained,
        abstention_correct=abstention_correct,
        claims=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate claim-evidence citation support")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()

    golden = load_golden(args.output, args.annotator)
    completed = {case.case_id for case in golden.cases}
    paths = sorted(args.answers.glob("*.json"))
    if not paths:
        raise ValueError(f"no answer bundles found in {args.answers}")
    for path in paths:
        case_id = path.stem.removesuffix(".answer")
        if case_id in completed:
            continue
        bundle = AnswerBundle.model_validate_json(path.read_text(encoding="utf-8"))
        golden.cases.append(annotate_bundle(case_id, bundle))
        save_atomic(args.output, golden)
        print(f"Saved {case_id}. Total cases: {len(golden.cases)}")
    print(f"Citation Golden Set: {args.output} ({len(golden.cases)} cases)")


if __name__ == "__main__":
    main()

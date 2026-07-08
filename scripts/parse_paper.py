import argparse
import json
from pathlib import Path

from paper_agent.ingestion import MinerUAdapter, paper_id_from_pdf


def find_content_list(root: Path, pdf_stem: str) -> Path:
    exact = list(root.rglob(f"{pdf_stem}_content_list.json"))
    if len(exact) == 1:
        return exact[0]
    if not exact:
        raise FileNotFoundError(
            f"cannot find {pdf_stem}_content_list.json below {root}"
        )
    raise ValueError(f"multiple content lists found for {pdf_stem}: {exact}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MinerU output to Paper JSON")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--mineru-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mineru-version", default="3.4.2")
    args = parser.parse_args()

    paper_id = paper_id_from_pdf(args.pdf)
    content_list = find_content_list(args.mineru_output, args.pdf.stem)
    paper = MinerUAdapter(
        content_list,
        version=f"mineru-{args.mineru_version}",
    ).parse(str(args.pdf), paper_id)

    output_dir = args.output / paper.paper_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "paper.json"
    output_file.write_text(
        json.dumps(paper.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"paper_id: {paper.paper_id}")
    print(f"title: {paper.title}")
    print(f"elements: {len(paper.elements)}")
    print(f"output: {output_file}")


if __name__ == "__main__":
    main()

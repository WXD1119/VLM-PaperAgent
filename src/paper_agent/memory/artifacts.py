from pathlib import Path

from pydantic import BaseModel

from paper_agent.domain import AnswerBundle


class AnswerArtifactSummary(BaseModel):
    """不可变回答工件的紧凑索引记录。"""

    path: str
    query: str
    answer_preview: str
    abstained: bool
    claim_count: int
    evidence_count: int
    generator_model: str | None = None


def load_answer_artifact(path: str | Path) -> AnswerBundle:
    return AnswerBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def list_answer_artifacts(root: str | Path = "artifacts/answers") -> list[AnswerArtifactSummary]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    summaries = []
    for path in sorted(root_path.glob("*.answer.json")):
        bundle = load_answer_artifact(path)
        answer_text = bundle.answer.answer.replace("\n", " ").strip()
        summaries.append(
            AnswerArtifactSummary(
                path=str(path),
                query=bundle.evidence_pack.query,
                answer_preview=answer_text[:160],
                abstained=bundle.answer.abstained,
                claim_count=len(bundle.answer.claims),
                evidence_count=len(bundle.evidence_pack.items),
                generator_model=bundle.generator_model,
            )
        )
    return summaries

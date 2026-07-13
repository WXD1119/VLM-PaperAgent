import importlib.util
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("batch judge tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
)


def load_batch_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "batch_judge_answers.py"
    spec = importlib.util.spec_from_file_location("batch_judge_answers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bundle(*, abstained: bool = False) -> AnswerBundle:
    claims = [] if abstained else [AnswerClaim(text="claim", evidence_ids=["E1"])]
    return AnswerBundle(
        evidence_pack=EvidencePack(
            query="q",
            items=[
                EvidenceItem(
                    evidence_id="E1",
                    chunk_id="c1",
                    paper_id="p1",
                    kind=ChunkKind.TEXT,
                    pages=[1],
                    content="evidence",
                )
            ],
        ),
        answer=GroundedAnswer(
            answer="answer" if not abstained else "insufficient",
            claims=claims,
            abstained=abstained,
            abstention_reason="insufficient evidence" if abstained else None,
        ),
        citation_validation=CitationValidation(
            valid=True,
            claim_count=len(claims),
            cited_claim_count=len(claims),
        ),
        generator_model="test",
    )


def write_answer(path: Path, answer: AnswerBundle) -> None:
    path.write_text(answer.model_dump_json(indent=2), encoding="utf-8")


def test_select_judge_jobs_skips_existing_and_abstained_by_default(tmp_path):
    batch = load_batch_module()
    answers = tmp_path / "answers"
    outputs = tmp_path / "evals"
    answers.mkdir()
    outputs.mkdir()
    write_answer(answers / "a.answer.json", bundle())
    write_answer(answers / "b.answer.json", bundle())
    write_answer(answers / "abstained.answer.json", bundle(abstained=True))
    (outputs / "a.glm-judge.json").write_text('{"assessments":[]}', encoding="utf-8")

    jobs = batch.select_judge_jobs(list(answers.glob("*.answer.json")), output_dir=outputs)

    assert [job.case_id for job in jobs] == ["b"]


def test_select_judge_jobs_supports_overwrite_include_abstained_and_limit(tmp_path):
    batch = load_batch_module()
    answers = tmp_path / "answers"
    outputs = tmp_path / "evals"
    answers.mkdir()
    outputs.mkdir()
    write_answer(answers / "a.answer.json", bundle())
    write_answer(answers / "abstained.answer.json", bundle(abstained=True))
    (outputs / "a.glm-judge.json").write_text('{"assessments":[]}', encoding="utf-8")

    jobs = batch.select_judge_jobs(
        list(answers.glob("*.answer.json")),
        output_dir=outputs,
        overwrite=True,
        include_abstained=True,
        limit=1,
    )

    assert [job.case_id for job in jobs] == ["a"]

import json
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("answer quality evaluation tests require pydantic v2", allow_module_level=True)

from paper_agent.domain import (
    AnswerBundle,
    AnswerClaim,
    ChunkKind,
    CitationValidation,
    ClaimSupportAssessment,
    EvidenceItem,
    EvidencePack,
    GroundedAnswer,
    SemanticCitationReport,
    SupportVerdict,
)
from paper_agent.evaluation import evaluate_answer_quality, evaluate_answer_quality_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_answer_quality.py"


def _bundle(query: str, *, claims: list[AnswerClaim], abstained: bool = False) -> AnswerBundle:
    return AnswerBundle(
        evidence_pack=EvidencePack(
            query=query,
            items=[
                EvidenceItem(
                    evidence_id="E1",
                    chunk_id="c1",
                    paper_id="p1",
                    kind=ChunkKind.TEXT,
                    pages=[1],
                    content="Evidence text",
                )
            ],
        ),
        answer=GroundedAnswer(
            answer="Answer" if not abstained else "Insufficient evidence.",
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


def test_evaluate_answer_quality_tracks_unsupported_claim_rate():
    answers = {
        "supported": _bundle(
            "q1",
            claims=[AnswerClaim(text="supported claim", evidence_ids=["E1"])],
        ),
        "unsupported": _bundle(
            "q2",
            claims=[AnswerClaim(text="unsupported claim", evidence_ids=["E1"])],
        ),
    }
    reports = {
        "supported": SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.SUPPORTED,
                    evidence_ids=["E1"],
                    reasoning_summary="ok",
                )
            ]
        ),
        "unsupported": SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.UNSUPPORTED,
                    evidence_ids=[],
                    reasoning_summary="not in evidence",
                )
            ]
        ),
    }

    result = evaluate_answer_quality(answers, reports)

    assert result.answer_count == 2
    assert result.claim_count == 2
    assert result.citation_pass_rate == 1.0
    assert result.semantic_coverage_rate == 1.0
    assert result.semantic_support_rate == 0.5
    assert result.unsupported_claim_rate == 0.5
    assert result.fully_supported_answer_rate == 0.5


def test_answer_quality_gate_reports_failures_and_risky_cases():
    answers = {
        "supported": _bundle(
            "q1",
            claims=[AnswerClaim(text="supported claim", evidence_ids=["E1"])],
        ),
        "missing_judge": _bundle(
            "q2",
            claims=[AnswerClaim(text="claim without judge", evidence_ids=["E1"])],
        ),
    }
    reports = {
        "supported": SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.SUPPORTED,
                    evidence_ids=["E1"],
                    reasoning_summary="ok",
                )
            ]
        )
    }

    result = evaluate_answer_quality(answers, reports)
    gate = evaluate_answer_quality_gate(
        result,
        min_semantic_coverage=1.0,
        min_semantic_support=1.0,
        max_unsupported_claim_rate=0.0,
        min_fully_supported_answer_rate=1.0,
    )

    assert not gate.passed
    assert {failure.metric for failure in gate.failures} == {
        "semantic_coverage_rate",
        "fully_supported_answer_rate",
    }
    assert gate.risky_cases == ["missing_judge"]


def test_evaluate_answer_quality_script_writes_json(tmp_path):
    answers_dir = tmp_path / "answers"
    judges_dir = tmp_path / "judges"
    output = tmp_path / "quality.json"
    answers_dir.mkdir()
    judges_dir.mkdir()
    (answers_dir / "case.answer.json").write_text(
        _bundle("q", claims=[AnswerClaim(text="claim", evidence_ids=["E1"])]).model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )
    (judges_dir / "case.glm-judge.json").write_text(
        SemanticCitationReport(
            assessments=[
                ClaimSupportAssessment(
                    claim_index=1,
                    verdict=SupportVerdict.SUPPORTED,
                    evidence_ids=["E1"],
                    reasoning_summary="ok",
                )
            ]
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--answers",
            str(answers_dir),
            "--judges",
            str(judges_dir),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "Unsupported claim rate: 0.0000" in completed.stdout
    assert payload["semantic_support_rate"] == 1.0
    assert payload["fully_supported_answer_rate"] == 1.0


def test_evaluate_answer_quality_script_quality_gate_fails(tmp_path):
    answers_dir = tmp_path / "answers"
    judges_dir = tmp_path / "judges"
    output = tmp_path / "quality.json"
    answers_dir.mkdir()
    judges_dir.mkdir()
    (answers_dir / "case.answer.json").write_text(
        _bundle("q", claims=[AnswerClaim(text="claim", evidence_ids=["E1"])]).model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--answers",
            str(answers_dir),
            "--judges",
            str(judges_dir),
            "--output",
            str(output),
            "--min-semantic-coverage",
            "1.0",
            "--fail-on-gate",
        ],
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert completed.returncode == 1
    assert "Quality gate: FAIL" in completed.stdout
    assert "semantic_coverage_rate" in completed.stdout
    assert output.exists()

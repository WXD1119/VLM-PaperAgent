import json
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("context guard evaluation tests require pydantic v2", allow_module_level=True)

from pydantic import TypeAdapter

from paper_agent.evaluation import ContextGuardCase, evaluate_context_guard


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "evals" / "context_guard.seed.json"
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_context_guard.py"


def test_context_guard_seed_golden_scores_perfectly():
    cases = TypeAdapter(list[ContextGuardCase]).validate_json(
        GOLDEN_PATH.read_text(encoding="utf-8")
    )

    result = evaluate_context_guard(cases)

    assert result.case_count == 6
    assert result.accuracy == 1.0
    assert result.constraint_recall == 1.0
    assert result.clarification_recall == 1.0
    assert result.wrong_constraint_rate == 0.0


def test_evaluate_context_guard_script_writes_json(tmp_path):
    output = tmp_path / "context_guard_eval.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--golden",
            str(GOLDEN_PATH),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "Accuracy: 1.0000" in completed.stdout
    assert payload["accuracy"] == 1.0
    assert payload["wrong_constraint_rate"] == 0.0

import importlib.util
from pathlib import Path

import pydantic
import pytest

if not hasattr(pydantic, "model_validator"):
    pytest.skip("GLM judge service tests require pydantic v2", allow_module_level=True)


def load_server_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve_glm_judge.py"
    spec = importlib.util.spec_from_file_location("serve_glm_judge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_json_repairs_unquoted_claim_assessment():
    server = load_server_module()

    value = server.extract_json(
        "<answer>{claim_index: 2, verdict: supported, evidence_ids: [E3], "
        "reasoning_summary: Evidence E3 directly supports the claim.}</answer>",
        {"title": "ClaimSupportAssessment"},
    )

    assert value["claim_index"] == 2
    assert value["verdict"] == "supported"
    assert value["evidence_ids"] == ["E3"]
    assert "Recovered from malformed judge JSON" in value["reasoning_summary"]


def test_extract_json_keeps_strict_json_path():
    server = load_server_module()

    value = server.extract_json(
        '<answer>{"claim_index": 1, "verdict": "unsupported", '
        '"evidence_ids": [], "reasoning_summary": "missing evidence"}</answer>',
        {"title": "ClaimSupportAssessment"},
    )

    assert value["claim_index"] == 1
    assert value["verdict"] == "unsupported"

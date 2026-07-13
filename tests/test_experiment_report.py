import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_experiment_report.py"


def test_build_experiment_report_writes_markdown(tmp_path):
    registry = tmp_path / "registry.json"
    answer_quality = tmp_path / "answer_quality.json"
    context_guard = tmp_path / "context_guard.json"
    output = tmp_path / "report.md"

    registry.write_text(
        json.dumps(
            [
                {
                    "experiment_id": "retrieval_ablation_v1_16",
                    "status": "completed",
                    "category": "retrieval_ablation",
                    "dataset": {"name": "retrieval_golden.v1.json", "cases": 16},
                    "metrics": {
                        "RRF_BGE_reranker_v2_m3": {
                            "hit_at_1": 1.0,
                            "hit_at_5": 1.0,
                            "recall_at_1": 0.5938,
                            "recall_at_5": 0.8984,
                            "mrr": 0.7969,
                            "ndcg_at_5": 0.8019,
                        }
                    },
                    "conclusion": "中文结论：Reranker wins.",
                }
            ]
        ),
        encoding="utf-8",
    )
    answer_quality.write_text(
        json.dumps(
            {
                "answer_count": 12,
                "claim_count": 37,
                "citation_pass_rate": 1.0,
                "semantic_coverage_rate": 1.0,
                "semantic_support_rate": 1.0,
                "unsupported_claim_rate": 0.0,
                "fully_supported_answer_rate": 0.8333,
            }
        ),
        encoding="utf-8",
    )
    context_guard.write_text(
        json.dumps(
            {
                "case_count": 4,
                "accuracy": 1.0,
                "constraint_recall": 1.0,
                "clarification_recall": 1.0,
                "wrong_constraint_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--registry",
            str(registry),
            "--answer-quality",
            str(answer_quality),
            "--context-guard",
            str(context_guard),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    report = output.read_text(encoding="utf-8")
    assert "output:" in completed.stdout
    assert "Retrieval Ablation" in report
    assert "RRF_BGE_reranker_v2_m3" in report
    assert "Hit@1" in report
    assert "中文结论" in report
    assert "Unsupported claim rate | 0.0%" in report
    assert "Quality gate result: PASS" in report
    assert "Resume-Ready Summary" in report

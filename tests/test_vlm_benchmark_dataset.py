import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "evals" / "vlm_benchmark_manifest.v2.json"
QUESTION_SCRIPT = REPO_ROOT / "scripts" / "build_vlm_benchmark_questions.py"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_vlm_papers.py"
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_vlm_benchmark.py"
INSPECT_SCRIPT = REPO_ROOT / "scripts" / "inspect_vlm_benchmark.py"
POSTPROCESS_SCRIPT = REPO_ROOT / "scripts" / "postprocess_vlm_benchmark.py"
PAPER_ID_MAP_SCRIPT = REPO_ROOT / "scripts" / "build_vlm_paper_id_map.py"


def test_vlm_benchmark_manifest_has_ten_papers_and_question_seeds():
    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(papers) == 10
    assert len({paper["paper_key"] for paper in papers}) == 10
    assert all(paper["pdf_url"].startswith("https://arxiv.org/pdf/") for paper in papers)
    assert sum(len(paper["question_seeds"]) for paper in papers) == 50
    assert {"method", "dataset", "evaluation"}.issubset(
        {
            seed["category"]
            for paper in papers
            for seed in paper["question_seeds"]
        }
    )


def test_build_vlm_benchmark_questions_script_writes_draft(tmp_path):
    output = tmp_path / "questions.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(QUESTION_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    questions = json.loads(output.read_text(encoding="utf-8"))
    assert "questions: 50" in completed.stdout
    assert len(questions) == 50
    assert all(item["annotation_status"] == "draft" for item in questions)
    assert all("expected_answer" in item for item in questions)


def test_download_vlm_papers_dry_run_lists_all_papers(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(DOWNLOAD_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "papers: 10" in completed.stdout
    assert "completed: 0" in completed.stdout
    assert "failed: 0" in completed.stdout


def test_prepare_vlm_benchmark_dry_run_prints_mineru_command(tmp_path):
    output_questions = tmp_path / "questions.json"
    mineru_output = tmp_path / "mineru"
    papers_output = tmp_path / "papers"

    completed = subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--pdf-dir",
            str(REPO_ROOT / "data" / "raw" / "vlm_benchmark"),
            "--mineru-output",
            str(mineru_output),
            "--papers-output",
            str(papers_output),
            "--questions-output",
            str(output_questions),
            "--print-mineru-command",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "papers: 10" in completed.stdout
    assert "MinerU command:" in completed.stdout
    assert "mineru -p" in completed.stdout
    assert "question_draft:" in completed.stdout
    assert "parse_jobs: 0" in completed.stdout


def test_inspect_vlm_benchmark_reports_pdf_status(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(INSPECT_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--pdf-dir",
            str(REPO_ROOT / "data" / "raw" / "vlm_benchmark"),
            "--mineru-output",
            str(tmp_path / "mineru"),
            "--papers-output",
            str(tmp_path / "papers"),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "papers: 10" in completed.stdout
    assert "pdf_ready:" in completed.stdout
    assert "chunks_ready:" in completed.stdout


def test_postprocess_vlm_benchmark_dry_run_prints_pipeline_commands(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(POSTPROCESS_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--pdf-dir",
            str(REPO_ROOT / "data" / "raw" / "vlm_benchmark"),
            "--papers",
            str(tmp_path / "papers"),
            "--db",
            str(tmp_path / "chroma"),
            "--graph-output",
            str(tmp_path / "graph"),
            "--eval-output-dir",
            str(tmp_path / "evals"),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "mode: dry-run" in completed.stdout
    assert "scripts/index_dense.py" in completed.stdout
    assert "scripts/build_graph.py" in completed.stdout
    assert "Annotation command:" in completed.stdout
    assert "scripts/annotate_retrieval.py" in completed.stdout
    assert "--paper-id-map" in completed.stdout
    assert "After annotation, evaluate:" in completed.stdout
    assert "scripts/evaluate_reranked.py" in completed.stdout


def test_build_vlm_paper_id_map_script_handles_missing_papers(tmp_path):
    output = tmp_path / "map.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PAPER_ID_MAP_SCRIPT),
            "--manifest",
            str(MANIFEST),
            "--papers",
            str(tmp_path / "papers"),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert "papers: 10" in completed.stdout
    assert "matched: 0" in completed.stdout
    assert len(rows) == 10
    assert all(row["status"] == "missing" for row in rows)

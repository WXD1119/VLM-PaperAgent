from pathlib import Path

import pytest

from paper_agent.ingestion import IngestionStatus, LocalIngestionTaskStore


def test_local_ingestion_task_store_persists_upload_and_retry(tmp_path: Path):
    store = LocalIngestionTaskStore(tmp_path / "ingestion")
    task = store.create("sample.pdf", b"%PDF-1.7")

    assert task.status == IngestionStatus.PENDING
    assert Path(task.source_path).read_bytes() == b"%PDF-1.7"
    assert store.load(task.task_id).filename == "sample.pdf"

    task.status = IngestionStatus.FAILED
    task.error = "parser failed"
    store.save(task)
    retried = store.retry(task.task_id)
    assert retried.status == IngestionStatus.PENDING
    assert retried.error is None


def test_local_ingestion_task_store_rejects_non_pdf(tmp_path: Path):
    with pytest.raises(ValueError, match="only PDF"):
        LocalIngestionTaskStore(tmp_path).create("notes.txt", b"not a PDF")

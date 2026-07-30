"""Sequential worker for durable ingestion tasks.

This is intentionally a small, inspectable first version. A future Celery/RQ
worker can reuse the same task record and status transitions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from paper_agent.ingestion.jobs import IngestionStatus, IngestionTask, LocalIngestionTaskStore
from paper_agent.ingestion.mineru_adapter import paper_id_from_pdf


class IngestionWorker:
    def __init__(
        self,
        store: LocalIngestionTaskStore,
        *,
        project_root: str | Path = ".",
        timeout_seconds: int = 1800,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.timeout_seconds = timeout_seconds
        self.papers_root = Path(os.getenv("PAPER_AGENT_PAPERS_ROOT", "artifacts/papers"))
        self.mineru_root = Path(os.getenv("PAPER_AGENT_MINERU_ROOT", "artifacts/mineru_web"))
        self.chroma_db = Path(os.getenv("PAPER_AGENT_CHROMA_DB", "artifacts/chroma"))
        self.collection = os.getenv("PAPER_AGENT_CHROMA_COLLECTION", "paper_chunks_bge_m3")
        self.embedding_model = os.getenv("PAPER_AGENT_EMBEDDING_MODEL", "artifacts/models/bge-m3")
        self.device = os.getenv("PAPER_AGENT_DEVICE") or None
        self.offline = os.getenv("PAPER_AGENT_OFFLINE", "1") not in {"0", "false", "False"}
        self.mineru_command = os.getenv("PAPER_AGENT_MINERU_COMMAND", "mineru").split()

    def run_next(self) -> IngestionTask | None:
        task = next((item for item in self.store.list(limit=1000) if item.status == IngestionStatus.PENDING), None)
        return self.run(task) if task else None

    def run(self, task: IngestionTask) -> IngestionTask:
        task.status = IngestionStatus.RUNNING
        task.stage = "mineru"
        task.logs.append("worker started")
        self.store.save(task)
        print(f"[{task.task_id}] started: {task.filename}", flush=True)
        try:
            source = Path(task.source_path).resolve()
            paper_id = paper_id_from_pdf(source)
            input_dir = self.store.root / "mineru_input" / task.task_id
            input_dir.mkdir(parents=True, exist_ok=True)
            staged_pdf = input_dir / task.filename
            staged_pdf.write_bytes(source.read_bytes())
            mineru_output = self.mineru_root / task.task_id

            self._run(task, [*self.mineru_command, "-p", str(input_dir), "-o", str(mineru_output), "-b", "pipeline"])
            task.stage = "normalize"
            self.store.save(task)
            self._run(
                task,
                [
                    sys.executable,
                    "scripts/parse_paper.py",
                    "--pdf",
                    str(staged_pdf),
                    "--mineru-output",
                    str(mineru_output),
                    "--output",
                    str(self.papers_root),
                ],
            )
            paper_path = self.papers_root / paper_id / "paper.json"
            task.stage = "chunk"
            self.store.save(task)
            self._run(task, [sys.executable, "scripts/chunk_paper.py", "--paper", str(paper_path)])
            chunks_path = paper_path.with_name("chunks.json")

            task.stage = "index"
            self.store.save(task)
            index_command = [
                sys.executable,
                "scripts/index_dense.py",
                "--chunks",
                str(self.papers_root),
                "--db",
                str(self.chroma_db),
                "--collection",
                self.collection,
                "--model",
                self.embedding_model,
            ]
            if self.device:
                index_command.extend(["--device", self.device])
            if self.offline:
                index_command.append("--offline")
            self._run(task, index_command)

            task.status = IngestionStatus.SUCCEEDED
            task.stage = "ready_for_promotion"
            task.paper_id = paper_id
            task.paper_path = str(paper_path)
            task.chunks_path = str(chunks_path)
            task.logs.append("ingestion completed; ready for workspace promotion")
            print(f"[{task.task_id}] completed: paper_id={paper_id}", flush=True)
        except Exception as exc:
            task.status = IngestionStatus.FAILED
            task.stage = "failed"
            task.error = str(exc)
            task.logs.append(f"failed: {exc}")
            print(f"[{task.task_id}] failed: {exc}", flush=True)
        return self.store.save(task)

    def _run(self, task: IngestionTask, command: list[str]) -> None:
        task.logs.append("$ " + " ".join(command))
        self.store.save(task)
        print(f"[{task.task_id}] stage={task.stage}", flush=True)
        print(f"[{task.task_id}] command: {' '.join(command)}", flush=True)
        result = subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        if output:
            task.logs.extend(output.splitlines()[-20:])
        if result.returncode != 0:
            raise RuntimeError(f"command exited {result.returncode}: {' '.join(command)}")
        print(f"[{task.task_id}] stage={task.stage} finished", flush=True)

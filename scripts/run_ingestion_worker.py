import argparse
import time
from pathlib import Path

from paper_agent.ingestion import LocalIngestionTaskStore
from paper_agent.ingestion.worker import IngestionWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run queued PDF ingestion tasks sequentially")
    parser.add_argument("--root", type=Path, default=Path("artifacts/ingestion"))
    parser.add_argument("--once", action="store_true", help="Process at most one pending task")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    worker = IngestionWorker(
        LocalIngestionTaskStore(args.root), timeout_seconds=args.timeout_seconds
    )
    while True:
        task = worker.run_next()
        if task is not None:
            print(f"task_id: {task.task_id}")
            print(f"status: {task.status}")
            print(f"stage: {task.stage}")
            if task.error:
                print(f"error: {task.error}")
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

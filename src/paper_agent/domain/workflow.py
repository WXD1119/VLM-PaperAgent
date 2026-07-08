from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowState(StrEnum):
    INIT = "init"
    PARSING = "parsing"
    INDEXING = "indexing"
    EXTRACTING_CLAIMS = "extracting_claims"
    RETRIEVING_EVIDENCE = "retrieving_evidence"
    CRITIQUING = "critiquing"
    JUDGING = "judging"
    REFLECTING = "reflecting"
    REPORTING = "reporting"
    HUMAN_REVIEW = "human_review"
    FINISHED = "finished"
    FAILED = "failed"


class RunContext(BaseModel):
    run_id: str
    paper_id: str
    pdf_path: str
    state: WorkflowState = WorkflowState.INIT
    attempt_by_state: dict[str, int] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    state_history: list[WorkflowState] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NodeResult(BaseModel):
    next_state: WorkflowState
    artifacts: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


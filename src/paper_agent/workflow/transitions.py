from paper_agent.domain.workflow import WorkflowState


TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.INIT: {WorkflowState.PARSING},
    WorkflowState.PARSING: {WorkflowState.INDEXING, WorkflowState.FAILED},
    WorkflowState.INDEXING: {WorkflowState.EXTRACTING_CLAIMS, WorkflowState.FAILED},
    WorkflowState.EXTRACTING_CLAIMS: {WorkflowState.RETRIEVING_EVIDENCE, WorkflowState.FAILED},
    WorkflowState.RETRIEVING_EVIDENCE: {WorkflowState.CRITIQUING, WorkflowState.FAILED},
    WorkflowState.CRITIQUING: {WorkflowState.JUDGING, WorkflowState.FAILED},
    WorkflowState.JUDGING: {
        WorkflowState.REFLECTING,
        WorkflowState.REPORTING,
        WorkflowState.HUMAN_REVIEW,
        WorkflowState.FAILED,
    },
    WorkflowState.REFLECTING: {
        WorkflowState.JUDGING,
        WorkflowState.HUMAN_REVIEW,
        WorkflowState.FAILED,
    },
    WorkflowState.REPORTING: {WorkflowState.FINISHED, WorkflowState.FAILED},
    WorkflowState.HUMAN_REVIEW: {WorkflowState.REPORTING, WorkflowState.FAILED},
    WorkflowState.FINISHED: set(),
    WorkflowState.FAILED: set(),
}


def validate_transition(current: WorkflowState, target: WorkflowState) -> None:
    if target not in TRANSITIONS[current]:
        raise ValueError(f"illegal workflow transition: {current} -> {target}")


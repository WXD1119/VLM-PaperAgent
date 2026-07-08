from dataclasses import dataclass

from paper_agent.domain.workflow import NodeResult, RunContext, WorkflowState
from paper_agent.workflow.checkpoint import InMemoryCheckpointStore
from paper_agent.workflow.scheduler import Scheduler


@dataclass
class Node:
    state: WorkflowState
    target: WorkflowState

    def execute(self, context: RunContext) -> NodeResult:
        return NodeResult(next_state=self.target, artifacts={self.state.value: True})


def test_scheduler_runs_happy_path_and_checkpoints() -> None:
    path = [
        (WorkflowState.PARSING, WorkflowState.INDEXING),
        (WorkflowState.INDEXING, WorkflowState.EXTRACTING_CLAIMS),
        (WorkflowState.EXTRACTING_CLAIMS, WorkflowState.RETRIEVING_EVIDENCE),
        (WorkflowState.RETRIEVING_EVIDENCE, WorkflowState.CRITIQUING),
        (WorkflowState.CRITIQUING, WorkflowState.JUDGING),
        (WorkflowState.JUDGING, WorkflowState.REPORTING),
        (WorkflowState.REPORTING, WorkflowState.FINISHED),
    ]
    store = InMemoryCheckpointStore()
    scheduler = Scheduler([Node(source, target) for source, target in path], store)
    context = RunContext(run_id="r1", paper_id="p1", pdf_path="paper.pdf")

    result = scheduler.run(context)

    assert result.state == WorkflowState.FINISHED
    assert store.load("r1").state == WorkflowState.FINISHED


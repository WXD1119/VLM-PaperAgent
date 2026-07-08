from typing import Protocol

from paper_agent.domain.workflow import NodeResult, RunContext, WorkflowState


class WorkflowNode(Protocol):
    state: WorkflowState

    def execute(self, context: RunContext) -> NodeResult: ...


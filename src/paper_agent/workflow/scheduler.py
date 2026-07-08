from datetime import datetime, timezone

from paper_agent.domain.workflow import RunContext, WorkflowState
from paper_agent.workflow.checkpoint import CheckpointStore
from paper_agent.workflow.node import WorkflowNode
from paper_agent.workflow.transitions import validate_transition


class Scheduler:
    def __init__(
        self,
        nodes: list[WorkflowNode],
        checkpoint_store: CheckpointStore,
        max_steps: int = 30,
        max_attempts: int = 2,
    ) -> None:
        self.nodes = {node.state: node for node in nodes}
        self.checkpoints = checkpoint_store
        self.max_steps = max_steps
        self.max_attempts = max_attempts

    def run(self, context: RunContext) -> RunContext:
        if context.state == WorkflowState.INIT:
            self._transition(context, WorkflowState.PARSING)

        steps = 0
        while context.state not in {WorkflowState.FINISHED, WorkflowState.FAILED}:
            steps += 1
            if steps > self.max_steps:
                return self._fail(context, "scheduler exceeded max_steps")

            node = self.nodes.get(context.state)
            if node is None:
                return self._fail(context, f"node is not registered for {context.state}")

            key = context.state.value
            context.attempt_by_state[key] = context.attempt_by_state.get(key, 0) + 1
            try:
                result = node.execute(context.model_copy(deep=True))
                context.artifacts.update(result.artifacts)
                self._transition(context, result.next_state)
            except Exception as exc:  # boundary: node failures are isolated here
                context.errors.append({"state": key, "error": str(exc)})
                if context.attempt_by_state[key] >= self.max_attempts:
                    return self._fail(context, f"node {key} exhausted retries")
                self.checkpoints.save(context)

        return context

    def _transition(self, context: RunContext, target: WorkflowState) -> None:
        validate_transition(context.state, target)
        context.state_history.append(context.state)
        context.state = target
        context.updated_at = datetime.now(timezone.utc)
        self.checkpoints.save(context)

    def _fail(self, context: RunContext, message: str) -> RunContext:
        context.errors.append({"state": context.state.value, "error": message})
        if context.state != WorkflowState.FAILED:
            context.state_history.append(context.state)
            context.state = WorkflowState.FAILED
        self.checkpoints.save(context)
        return context


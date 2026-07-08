from typing import Protocol

from paper_agent.domain.workflow import RunContext


class CheckpointStore(Protocol):
    def save(self, context: RunContext) -> None: ...
    def load(self, run_id: str) -> RunContext | None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, RunContext] = {}

    def save(self, context: RunContext) -> None:
        self._items[context.run_id] = context.model_copy(deep=True)

    def load(self, run_id: str) -> RunContext | None:
        item = self._items.get(run_id)
        return item.model_copy(deep=True) if item else None


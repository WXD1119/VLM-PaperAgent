from typing import Protocol

from paper_agent.domain.paper import PaperElement


class VectorStore(Protocol):
    def upsert(self, elements: list[PaperElement]) -> None: ...
    def search(self, query: str, top_k: int) -> list[str]: ...


class GraphStore(Protocol):
    def upsert_paper_relations(self, paper_id: str, relations: list[dict]) -> None: ...


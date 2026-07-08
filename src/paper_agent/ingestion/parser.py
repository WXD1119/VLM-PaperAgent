from typing import Protocol

from paper_agent.domain.paper import Paper


class PaperParser(Protocol):
    @property
    def version(self) -> str: ...
    def parse(self, pdf_path: str, paper_id: str) -> Paper: ...


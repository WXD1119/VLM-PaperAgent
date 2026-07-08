from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChunkKind(StrEnum):
    TEXT = "text"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"


class ParentChunk(BaseModel):
    parent_chunk_id: str
    paper_id: str
    section_path: list[str]
    content: str
    element_ids: list[str] = Field(min_length=1)
    pages: list[int] = Field(min_length=1)


class RetrievalChunk(BaseModel):
    chunk_id: str
    parent_chunk_id: str
    paper_id: str
    kind: ChunkKind
    content: str = Field(min_length=1)
    context: str = ""
    element_ids: list[str] = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    section_path: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        section = " > ".join(self.section_path)
        parts = [f"Section: {section}" if section else "", self.context, self.content]
        return "\n\n".join(part for part in parts if part)


class ChunkBundle(BaseModel):
    paper_id: str
    parents: list[ParentChunk]
    children: list[RetrievalChunk]


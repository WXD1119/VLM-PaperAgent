from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ElementType(StrEnum):
    PARAGRAPH = "paragraph"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"


class PaperElement(BaseModel):
    element_id: str
    paper_id: str
    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    section_path: list[str] = Field(default_factory=list)
    element_type: ElementType
    content: str = Field(min_length=1)
    parent_id: str | None = None
    prev_id: str | None = None
    next_id: str | None = None
    parser_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Paper(BaseModel):
    paper_id: str
    title: str
    source_path: str
    sha256: str
    elements: list[PaperElement] = Field(default_factory=list)

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
    bbox: tuple[float, float, float, float] | None = None  # 元素在页面中的位置框，格式为 (x0, y0, x1, y1)，通常表示左上角和右下角坐标；没有位置信息时为 None。
    section_path: list[str] = Field(default_factory=list)  # 元素所属的章节路径，例如 ["3 Experiments", "3.1 Dataset"]。
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
    sha256: str # 论文的SHA256哈希值，常用于去重，避免重复导入同一篇论文，是哈希算法，不是加密算法，不能通过哈希值还原原文件
    elements: list[PaperElement] = Field(default_factory=list) # 论文中的所有元素列表；Field 是 Pydantic 提供的函数，用来给模型字段设置：默认值等

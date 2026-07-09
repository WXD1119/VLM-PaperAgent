from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    PAPER = "Paper"
    SECTION = "Section"
    CHUNK = "Chunk"
    ANSWER = "Answer"
    CLAIM = "Claim"
    QUERY = "Query"


class EdgeType(StrEnum):
    CONTAINS = "CONTAINS"
    HAS_SECTION = "HAS_SECTION"
    HAS_CHUNK = "HAS_CHUNK"
    ASKED = "ASKED"
    ANSWERED_WITH = "ANSWERED_WITH"
    HAS_CLAIM = "HAS_CLAIM"
    SUPPORTED_BY = "SUPPORTED_BY"
    CITES = "CITES"


class GraphNode(BaseModel):
    node_id: str = Field(min_length=1)
    node_type: NodeType
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: EdgeType
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphDocument(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

"""Lightweight evidence graph models, builders and JSONL storage."""

from .builder import GraphBuilder
from .fragments import (
    build_answer_fragment,
    build_paper_fragment,
    build_papers_fragment,
    delta_from_fragment,
)
from .jsonl_store import read_graph_jsonl, write_graph_jsonl
from .model import EdgeType, GraphDocument, GraphEdge, GraphNode, NodeType
from .query import ConceptNeighborhood, GraphQuery
from .validation import GraphValidationReport, GraphValidator
from .workspace import (
    GraphDiff,
    GraphCommit,
    GraphDelta,
    GraphWorkspace,
    LocalGraphWorkspaceStore,
    apply_delta,
    diff_graphs,
)

__all__ = [
    "EdgeType",
    "GraphBuilder",
    "ConceptNeighborhood",
    "GraphDocument",
    "GraphEdge",
    "GraphNode",
    "GraphQuery",
    "GraphDiff",
    "GraphValidationReport",
    "GraphValidator",
    "GraphCommit",
    "GraphDelta",
    "GraphWorkspace",
    "LocalGraphWorkspaceStore",
    "NodeType",
    "build_answer_fragment",
    "build_paper_fragment",
    "build_papers_fragment",
    "delta_from_fragment",
    "apply_delta",
    "diff_graphs",
    "read_graph_jsonl",
    "write_graph_jsonl",
]

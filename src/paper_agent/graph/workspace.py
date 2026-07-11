from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from paper_agent.graph.builder import stable_id
from paper_agent.graph.jsonl_store import read_graph_jsonl
from paper_agent.graph.model import GraphDocument, GraphEdge, GraphNode


class GraphWorkspace(BaseModel):
    workspace_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_graph_path: str = Field(min_length=1)
    head_commit_id: str | None = None
    visibility: str = "private"
    created_at: str
    updated_at: str


class GraphCommit(BaseModel):
    commit_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    parent_commit_id: str | None = None
    author_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    created_at: str
    stats: dict[str, int] = Field(default_factory=dict)


class GraphDelta(BaseModel):
    added_nodes: list[GraphNode] = Field(default_factory=list)
    added_edges: list[GraphEdge] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)
    removed_edge_ids: list[str] = Field(default_factory=list)


class GraphDiff(BaseModel):
    added_node_ids: list[str] = Field(default_factory=list)
    added_edge_ids: list[str] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)
    removed_edge_ids: list[str] = Field(default_factory=list)
    added_node_types: dict[str, int] = Field(default_factory=dict)
    added_edge_types: dict[str, int] = Field(default_factory=dict)
    removed_node_types: dict[str, int] = Field(default_factory=dict)
    removed_edge_types: dict[str, int] = Field(default_factory=dict)


class LocalGraphWorkspaceStore:
    """File-backed graph workspace store for Git-like graph forks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create_fork(
        self,
        *,
        base_graph_path: str | Path,
        workspace_id: str,
        owner_id: str,
        name: str | None = None,
        message: str = "fork workspace",
    ) -> GraphWorkspace:
        base_path = Path(base_graph_path)
        _ensure_graph_files_exist(base_path)
        self.root.mkdir(parents=True, exist_ok=False)
        now = _now()
        commit_id = commit_id_for(workspace_id, None, message, now)
        workspace = GraphWorkspace(
            workspace_id=workspace_id,
            owner_id=owner_id,
            name=name or workspace_id,
            base_graph_path=str(base_path),
            head_commit_id=commit_id,
            created_at=now,
            updated_at=now,
        )
        self._write_workspace(workspace)
        self.write_commit(
            GraphCommit(
                commit_id=commit_id,
                workspace_id=workspace_id,
                parent_commit_id=None,
                author_id=owner_id,
                message=message,
                created_at=now,
                stats={
                    "added_nodes": 0,
                    "added_edges": 0,
                    "removed_nodes": 0,
                    "removed_edges": 0,
                },
            ),
            GraphDelta(),
        )
        return workspace

    def read_workspace(self) -> GraphWorkspace:
        return GraphWorkspace.model_validate_json(
            (self.root / "workspace.json").read_text(encoding="utf-8")
        )

    def write_commit(self, commit: GraphCommit, delta: GraphDelta) -> None:
        commit_dir = self._commit_dir(commit.commit_id)
        commit_dir.mkdir(parents=True, exist_ok=False)
        (commit_dir / "commit.json").write_text(
            commit.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _write_jsonl(commit_dir / "delta_nodes.jsonl", delta.added_nodes)
        _write_jsonl(commit_dir / "delta_edges.jsonl", delta.added_edges)
        _write_lines(commit_dir / "removed_nodes.jsonl", delta.removed_node_ids)
        _write_lines(commit_dir / "removed_edges.jsonl", delta.removed_edge_ids)

    def commit_delta(
        self,
        delta: GraphDelta,
        *,
        author_id: str,
        message: str,
    ) -> GraphCommit:
        workspace = self.read_workspace()
        now = _now()
        commit = GraphCommit(
            commit_id=commit_id_for(workspace.workspace_id, workspace.head_commit_id, message, now),
            workspace_id=workspace.workspace_id,
            parent_commit_id=workspace.head_commit_id,
            author_id=author_id,
            message=message,
            created_at=now,
            stats={
                "added_nodes": len(delta.added_nodes),
                "added_edges": len(delta.added_edges),
                "removed_nodes": len(delta.removed_node_ids),
                "removed_edges": len(delta.removed_edge_ids),
            },
        )
        self.write_commit(commit, delta)
        workspace.head_commit_id = commit.commit_id
        workspace.updated_at = now
        self._write_workspace(workspace)
        return commit

    def preview_delta(self, delta: GraphDelta) -> GraphDocument:
        return apply_delta(self.load_effective_graph(), delta)

    def read_commit(self, commit_id: str) -> tuple[GraphCommit, GraphDelta]:
        commit_dir = self._commit_dir(commit_id)
        commit = GraphCommit.model_validate_json(
            (commit_dir / "commit.json").read_text(encoding="utf-8")
        )
        delta = GraphDelta(
            added_nodes=_read_jsonl(commit_dir / "delta_nodes.jsonl", GraphNode),
            added_edges=_read_jsonl(commit_dir / "delta_edges.jsonl", GraphEdge),
            removed_node_ids=_read_lines(commit_dir / "removed_nodes.jsonl"),
            removed_edge_ids=_read_lines(commit_dir / "removed_edges.jsonl"),
        )
        return commit, delta

    def commit_chain(self) -> list[GraphCommit]:
        workspace = self.read_workspace()
        if workspace.head_commit_id is None:
            return []
        commits_by_id: dict[str, GraphCommit] = {}
        for path in sorted((self.root / "commits").glob("*/commit.json")):
            commit = GraphCommit.model_validate_json(path.read_text(encoding="utf-8"))
            commits_by_id[commit.commit_id] = commit

        chain: list[GraphCommit] = []
        current_id: str | None = workspace.head_commit_id
        seen: set[str] = set()
        while current_id is not None:
            if current_id in seen:
                raise ValueError(f"workspace commit cycle detected: {current_id}")
            seen.add(current_id)
            commit = commits_by_id.get(current_id)
            if commit is None:
                raise FileNotFoundError(f"missing workspace commit: {current_id}")
            chain.append(commit)
            current_id = commit.parent_commit_id
        return list(reversed(chain))

    def load_effective_graph(self) -> GraphDocument:
        workspace = self.read_workspace()
        graph = read_graph_jsonl(workspace.base_graph_path)
        for commit in self.commit_chain():
            _, delta = self.read_commit(commit.commit_id)
            graph = apply_delta(graph, delta)
        return graph

    def _write_workspace(self, workspace: GraphWorkspace) -> None:
        (self.root / "workspace.json").write_text(
            workspace.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    def _commit_dir(self, commit_id: str) -> Path:
        return self.root / "commits" / commit_id


def commit_id_for(
    workspace_id: str,
    parent_commit_id: str | None,
    message: str,
    created_at: str,
) -> str:
    return "gc_" + stable_id(workspace_id, parent_commit_id or "root", message, created_at)


def apply_delta(graph: GraphDocument, delta: GraphDelta) -> GraphDocument:
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {edge.edge_id: edge for edge in graph.edges}

    for node_id in delta.removed_node_ids:
        nodes.pop(node_id, None)
    for edge_id in delta.removed_edge_ids:
        edges.pop(edge_id, None)

    for node in delta.added_nodes:
        if node.node_id not in delta.removed_node_ids:
            nodes[node.node_id] = node
    for edge in delta.added_edges:
        if edge.edge_id not in delta.removed_edge_ids:
            edges[edge.edge_id] = edge

    filtered_edges = {
        edge_id: edge
        for edge_id, edge in edges.items()
        if edge.source_id in nodes and edge.target_id in nodes
    }
    return GraphDocument(
        nodes=sorted(nodes.values(), key=lambda node: node.node_id),
        edges=sorted(filtered_edges.values(), key=lambda edge: edge.edge_id),
    )


def diff_graphs(base: GraphDocument, target: GraphDocument) -> GraphDiff:
    base_nodes = {node.node_id: node for node in base.nodes}
    target_nodes = {node.node_id: node for node in target.nodes}
    base_edges = {edge.edge_id: edge for edge in base.edges}
    target_edges = {edge.edge_id: edge for edge in target.edges}

    added_node_ids = sorted(set(target_nodes) - set(base_nodes))
    removed_node_ids = sorted(set(base_nodes) - set(target_nodes))
    added_edge_ids = sorted(set(target_edges) - set(base_edges))
    removed_edge_ids = sorted(set(base_edges) - set(target_edges))

    return GraphDiff(
        added_node_ids=added_node_ids,
        added_edge_ids=added_edge_ids,
        removed_node_ids=removed_node_ids,
        removed_edge_ids=removed_edge_ids,
        added_node_types=_node_type_counts(target_nodes, added_node_ids),
        added_edge_types=_edge_type_counts(target_edges, added_edge_ids),
        removed_node_types=_node_type_counts(base_nodes, removed_node_ids),
        removed_edge_types=_edge_type_counts(base_edges, removed_edge_ids),
    )


def _node_type_counts(nodes: dict[str, GraphNode], node_ids: list[str]) -> dict[str, int]:
    return dict(Counter(nodes[node_id].node_type.value for node_id in node_ids))


def _edge_type_counts(edges: dict[str, GraphEdge], edge_ids: list[str]) -> dict[str, int]:
    return dict(Counter(edges[edge_id].edge_type.value for edge_id in edge_ids))


def _ensure_graph_files_exist(path: Path) -> None:
    missing = [
        str(candidate)
        for candidate in [path / "nodes.jsonl", path / "edges.jsonl"]
        if not candidate.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing graph JSONL files: {', '.join(missing)}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_jsonl(path: Path, values: list[BaseModel]) -> None:
    path.write_text(
        "".join(value.model_dump_json() + "\n" for value in values),
        encoding="utf-8",
    )


def _read_jsonl(path: Path, model: type[BaseModel]) -> list[Any]:
    if not path.exists():
        return []
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(value + "\n" for value in values), encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

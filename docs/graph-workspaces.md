# Graph Workspaces and Branching Design

This document describes how the current lightweight evidence graph can evolve into a
multi-user graph workspace system. The goal is to let a user either build a private
paper graph from scratch or fork an existing graph and continue from it, without
mutating the original graph.

## Motivation

The current graph records papers, sections, chunks, queries, answers, claims and
claim-to-evidence links in JSONL files. This is enough for a single-user offline demo,
but a deployed product needs stronger isolation:

- a public or team graph should remain stable after it is published;
- each user should be able to add papers, answers and annotations privately;
- users should be able to fork a graph in a Git-like way and experiment safely;
- graph changes should be auditable, diffable and mergeable later.

The design keeps JSONL as the source-of-truth format for the MVP. Neo4j can still be
introduced later as a query/visualization backend, but it should be treated as a
materialized view rather than the only copy of the graph.

## Scope

In scope:

- workspace metadata;
- immutable graph commits;
- copy-on-write graph deltas;
- fork, diff and effective-graph query semantics;
- validation rules that prove a user branch does not corrupt its base graph.

Out of scope for the first implementation:

- full authentication and billing;
- real-time collaborative editing;
- automatic conflict resolution during merge;
- replacing the JSONL graph store with Neo4j.

## Core Concepts

### Workspace

A `GraphWorkspace` is the user's current graph view. It can be empty or forked from
another workspace.

Suggested fields:

```json
{
  "workspace_id": "ws_...",
  "owner_id": "user_...",
  "name": "My VLM Reading Graph",
  "base_workspace_id": "ws_public_vlm",
  "head_commit_id": "gc_...",
  "visibility": "private",
  "created_at": "2026-07-10T00:00:00Z",
  "updated_at": "2026-07-10T00:00:00Z"
}
```

### Commit

A `GraphCommit` is an immutable change set. It is intentionally close to a Git commit:
one parent, a message, an author and a delta payload.

Suggested fields:

```json
{
  "commit_id": "gc_...",
  "workspace_id": "ws_...",
  "parent_commit_id": "gc_parent",
  "author_id": "user_...",
  "message": "Add BLIP-2 paper and Q-Former answer bundle",
  "delta_ref": "commits/gc_...",
  "stats": {
    "added_nodes": 18,
    "added_edges": 31,
    "removed_nodes": 0,
    "removed_edges": 0
  },
  "created_at": "2026-07-10T00:00:00Z"
}
```

### Delta

A `GraphDelta` contains only the user's changes. Existing base graph files are not
rewritten.

```text
delta_nodes.jsonl
delta_edges.jsonl
removed_nodes.jsonl
removed_edges.jsonl
```

For the MVP, deletion should use tombstones instead of physically deleting records.
This makes rollback, audit and diff operations much easier.

## Storage Layout

One possible local layout:

```text
artifacts/graph_workspaces/
  workspaces/
    ws_public_vlm/
      workspace.json
      commits/
        gc_root/
          delta_nodes.jsonl
          delta_edges.jsonl
          removed_nodes.jsonl
          removed_edges.jsonl
    ws_user_wxd_blip2/
      workspace.json
      commits/
        gc_001/
          delta_nodes.jsonl
          delta_edges.jsonl
          removed_nodes.jsonl
          removed_edges.jsonl
```

For deployment, the same logical layout can be moved to object storage or a database.
The important property is not the directory structure; it is that base commits remain
immutable and user changes are stored as overlays.

## Effective Graph Algorithm

The effective graph is the graph a user actually sees.

1. Start from the root/base workspace.
2. Collect the commit chain from root to the selected workspace head.
3. Load nodes and edges in commit order.
4. Apply tombstones for removed node IDs and edge IDs.
5. Drop edges whose source or target node has been removed.
6. Run the graph validator on the final effective graph.

Pseudo-code:

```python
def load_effective_graph(workspace_id: str) -> GraphDocument:
    commits = resolve_commit_chain(workspace_id)
    nodes = {}
    edges = {}
    removed_nodes = set()
    removed_edges = set()

    for commit in commits:
        delta = load_delta(commit)
        removed_nodes.update(delta.removed_node_ids)
        removed_edges.update(delta.removed_edge_ids)

        for node in delta.added_nodes:
            if node.id not in removed_nodes:
                nodes[node.id] = node

        for edge in delta.added_edges:
            if edge.id not in removed_edges:
                edges[edge.id] = edge

    nodes = {node_id: node for node_id, node in nodes.items() if node_id not in removed_nodes}
    edges = {
        edge_id: edge
        for edge_id, edge in edges.items()
        if edge_id not in removed_edges
        and edge.source in nodes
        and edge.target in nodes
    }

    graph = GraphDocument(nodes=list(nodes.values()), edges=list(edges.values()))
    GraphValidator().validate(graph).raise_if_invalid()
    return graph
```

## Operations

### Create an empty workspace

Creates a private root workspace with no base.

```text
create_workspace(owner_id, name, base_workspace_id=None)
```

### Fork a workspace

Creates a new workspace whose base is another workspace's current head. The fork does
not copy all graph files immediately; it records a base pointer and starts with an
empty delta.

```text
fork_workspace(base_workspace_id, owner_id, name)
```

### Commit a delta

Appends new graph changes to the current workspace. Typical deltas come from parsing a
new paper, saving an answer bundle, editing labels or adding manual notes.

```text
commit_delta(workspace_id, delta, message)
```

### Query a workspace

Loads the effective graph and then reuses the existing `GraphQuery` logic.

```text
query_effective_graph(workspace_id, query)
```

### Diff two workspaces

Compares effective graphs or commit chains to show added/removed papers, chunks,
answers and claim support edges.

```text
diff_workspaces(left_workspace_id, right_workspace_id)
```

### Merge a workspace

Merging should be a later feature. The MVP can support manual export/import first.
When implemented, merge should detect conflicts instead of silently overwriting data.

Conflict examples:

- two branches edit the same human label differently;
- a branch deletes a chunk that another branch cites;
- two branches create different nodes with the same semantic identity but different IDs.

## Correctness Invariants

Every workspace-level operation must preserve these rules:

1. A fork never rewrites the base workspace.
2. A commit's parent must exist.
3. Node IDs are unique in the effective graph.
4. Edge IDs are unique in the effective graph.
5. Every edge source and target exists.
6. A `Paper` node must link to at least one `Chunk`.
7. A non-abstained `Claim` must link to at least one supporting `Chunk`.
8. `SUPPORTED_BY` edges must target `Chunk` nodes.
9. Tombstoning a node also hides its incident edges in the effective graph.
10. A private workspace must not become visible to other users unless explicitly shared.

These invariants are the main difference between a toy graph dump and an auditable
paper-agent graph system.

## API Plan

Possible HTTP endpoints:

```text
POST /workspaces
POST /workspaces/{workspace_id}/fork
GET  /workspaces/{workspace_id}
GET  /workspaces/{workspace_id}/graph
GET  /workspaces/{workspace_id}/diff?against=...
POST /workspaces/{workspace_id}/papers
POST /workspaces/{workspace_id}/answers
POST /workspaces/{workspace_id}/commit
```

For the first version, the API can remain localhost-only and file-backed. A later web
deployment can add authentication, access control and a persistent database without
changing the graph semantics.

## Relationship to Neo4j

Neo4j is not "for Java"; the name comes from the Java ecosystem historically, but it is
a general graph database with Cypher query support and Python clients.

In this project:

- JSONL graph files are the reproducible source of truth.
- Workspace deltas provide Git-like branching semantics.
- Neo4j, if introduced, should be a derived index for visualization and complex graph
  queries.

This keeps the MVP lightweight while preserving an upgrade path to production-grade
graph storage.

## Implementation Phases

### P0: Current state

Single JSONL graph built from parsed papers and answer bundles.

### P1: Workspace metadata and effective graph reader

Add workspace and commit models, local workspace store, fork command and effective
graph loading.

Deliverables:

- `paper_agent.graph.workspace`
- `scripts/fork_graph.py`
- `scripts/inspect_workspace.py`
- tests for fork isolation and effective graph validation.

Current P1 implementation:

- `GraphWorkspace`, `GraphCommit` and `GraphDelta` are file-backed Pydantic models.
- `LocalGraphWorkspaceStore.create_fork()` creates a workspace with a base graph pointer
  and an empty root commit.
- `LocalGraphWorkspaceStore.load_effective_graph()` applies workspace commits over the
  base graph and hides tombstoned nodes/edges.
- `scripts/fork_graph.py` creates local workspace forks.
- `scripts/inspect_workspace.py` validates the workspace effective graph.
- Tests cover fork isolation, base graph immutability, effective graph loading and
  tombstone behavior.

### P2: Delta-aware graph writing

Make paper ingestion and answer saving write deltas into a selected workspace instead
of rebuilding only one global graph.

Deliverables:

- `scripts/commit_graph_delta.py`
- `scripts/diff_graph.py`
- tests for added papers, added answers and tombstone behavior.

Current P2 implementation:

- `GraphDelta` can add nodes, add edges, hide nodes and hide edges.
- `LocalGraphWorkspaceStore.commit_delta()` appends immutable delta commits and advances
  the workspace head.
- `LocalGraphWorkspaceStore.preview_delta()` simulates a delta before writing it.
- `diff_graphs()` compares a base graph and a workspace effective graph by node/edge ID
  and summarizes added/removed node and edge types.
- `scripts/commit_graph_delta.py` commits JSONL node/edge deltas after validation.
- `scripts/diff_graph.py` reports the difference between a base graph and a workspace
  effective graph.
- Tests cover added records, base graph immutability, diff summaries and invalid delta
  detection.

### P3: Workspace-aware API

Expose workspace IDs through `/ask` and graph endpoints.

Deliverables:

- `POST /workspaces`
- `POST /workspaces/{id}/fork`
- `POST /workspaces/{id}/ask`
- `GET /workspaces/{id}/graph`

### P4: Optional Neo4j materialized view

Import effective graphs into Neo4j for visualization and Cypher demos. This should be
optional, because the resume-critical value is traceability and isolation, not the
presence of a heavyweight database.

## Resume Value

This design lets the project claim more than "RAG over PDFs":

- evidence graph with claim-level provenance;
- Git-like graph workspace branching;
- immutable deltas and audit trail;
- multi-user isolation by design;
- JSONL-first reproducibility with an optional Neo4j upgrade path.

That combination is technically richer than a simple vector database demo while still
being realistic enough to implement before autumn recruiting.

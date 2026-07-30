# Product Service Plan

This document defines what the paper reading agent provides to users, and separates
user-facing services from lower-level system modules. This separation is important:
some modules store or validate graph data, while user-facing services turn that graph
into answers, exploration panels, or workspace operations.

## Product Positioning

The project is an evidence-centered paper reading workspace. Users upload papers, ask
questions, inspect evidence, explore concepts, and maintain personal graph branches.

The key promise is:

```text
answer -> claim -> evidence chunk -> paper / section / page
```

The system should not only answer questions, but also show why the answer is supported
by the paper.

## Lower-level System Modules

These are implementation building blocks. They are not the same thing as user-facing
features.

| Module | Responsibility |
| --- | --- |
| Paper Parser | Converts PDFs into normalized `paper.json` with text, equations, tables, figures and section paths. |
| Chunker | Converts normalized paper elements into retrievable and citable chunks. |
| Retrieval Index | Provides BM25, dense retrieval, hybrid RRF and reranking. |
| Paper Knowledge Graph | Stores paper-only content such as `Paper`, `Section`, `Chunk`, `Concept` and later paper-extracted claims/methods/datasets/metrics. |
| Agent Memory | Stores session state, episodic event history, user profile preferences and answer/evaluation artifacts outside the paper graph. |
| Citation Judge | Checks citation integrity and semantic claim-evidence support. |
| Workspace Store | Stores graph workspace metadata, commits, deltas and fork relationships. |
| Neo4j Exporter | Later materializes the effective graph into Neo4j for Cypher query and visualization. |

`Paper Knowledge Graph` is the graph foundation. It can answer structural graph queries
about paper content, but it is not the complete user experience by itself. User profile,
conversation history and generated answers belong to Agent Memory or artifacts, not the
paper graph.

## User-facing Services

### 1. Paper Library

Users upload and manage papers.

Typical actions:

- upload a PDF;
- inspect parsing status;
- view title, pages and sections;
- add a parsed paper to the evidence graph.

Output style: library table and paper metadata.

### 2. Evidence Retrieval

Users ask for relevant evidence chunks.

Typical inputs:

```text
Find evidence about Q-Former and frozen LLM.
```

Output style: ranked evidence cards with paper, page, section, chunk type and content.

### 3. Evidence-grounded QA

Users ask a natural-language question. The system retrieves evidence and generates an
answer constrained by the current evidence pack.

Typical input:

```text
How does Q-Former bridge the frozen image encoder and frozen language model?
```

Output style: natural-language answer with claim-level citations.

Example:

```text
Q-Former bridges the frozen image encoder and frozen language model by acting as a
trainable module that extracts visual features with learnable query embeddings and
conditions the LLM on those features. [E1][E2]
```

### 4. Citation Verification

Users or the runtime system verify whether the answer's claims are actually supported
by the cited evidence.

Output style: validation report.

Example:

```text
claim=1 verdict=supported evidence=[E1]
claim=2 verdict=partially_supported evidence=[E2]
all_supported=false
```

### 5. Graph-grounded QA

This is a question-answering mode that uses the evidence graph, not only flat retrieved
chunks. It can follow graph edges such as:

```text
Concept -> Claim -> SUPPORTED_BY -> Chunk -> Section -> Paper
```

Input style: a question.

Example input:

```text
What role does Q-Former play in BLIP-2?
```

Output style: a natural-language answer with citations and optional graph-derived
context.

Example output:

```text
Q-Former is the trainable bridge between the frozen image encoder and the frozen LLM in
BLIP-2. It extracts a fixed number of visual features with learnable query embeddings,
then later projects query outputs into the LLM embedding space as soft visual prompts.
[E1][E2]
```

This service is still QA. The user expects an explanation paragraph, not a structured
exploration panel.

### 6. Concept Graph Explorer

This is an exploration mode. The user enters a keyword or concept, and the system
returns a structured graph neighborhood instead of only an answer paragraph.

Input style: a keyword or concept.

Example input:

```text
Q-Former
```

Output style: structured concept panel and local graph.

Example output:

```text
Concept: Q-Former

Related Papers:
- BLIP-2

Related Sections:
- 3.1 Model Architecture
- 3.3 Bootstrap Vision-to-Language Generative Learning

Related Claims:
1. Q-Former bridges a frozen image encoder and a frozen LLM.
   Evidence: BLIP-2, page 2, section 3.1

2. Q-Former uses learnable query embeddings and cross-attention.
   Evidence: BLIP-2, page 2, section 3.1

Related Concepts:
- frozen image encoder
- frozen LLM
- learnable query embeddings
- cross-attention
- soft visual prompts
```

Concept Graph Explorer and Graph-grounded QA both use the graph, but their outputs are
different:

```text
Graph-grounded QA: question -> paragraph answer
Concept Graph Explorer: keyword -> structured graph panel
```

### 7. Workspace and Branch Management

Users manage personal graph branches.

Typical actions:

- create a graph workspace from scratch;
- fork a public/team graph;
- add private papers;
- diff the private graph against its base;
- export the effective graph to Neo4j when visualization or Cypher queries are needed.

Output style: workspace list, diff report and validation status.

The boundary is intentional. Ordinary questions, generated answers, user preferences and
session state remain in Agent Memory or saved artifacts. The paper graph stays focused on
paper content.

## GraphRAG Layering

Both Graph-grounded QA and Concept Graph Explorer are GraphRAG features, but they use
GraphRAG differently.

```text
Paper Knowledge Graph = external paper-content knowledge store
Agent Memory = conversation/task/user memory outside the graph
Graph-grounded QA = GraphRAG answer mode
Concept Graph Explorer = GraphRAG exploration mode
```

Plain RAG retrieves chunks directly. GraphRAG retrieves or expands through graph
relationships before generating an answer or presenting a graph neighborhood.

## Near-term Implementation Plan

1. Keep the current evidence-grounded QA and citation judge stable.
2. Add `Concept` nodes and `MENTIONS` edges to the graph builder. Done in the first
   deterministic implementation; mention edges keep context previews and source metadata for
   later disambiguation.
3. Add keyword graph search over `Concept` and `Chunk`. The first version uses
   `query_graph.py --concept` and returns candidate concepts when a keyword is ambiguous.
4. Extend `query_graph.py --concept` into a fuller Concept Graph Explorer panel that prints the structured
   Concept Graph Explorer panel.
5. Implement graph workspace fork/effective graph loading.
6. Neo4j export is available as an optional materialized view; keep JSONL/workspace as
   the source of truth and use Neo4j for visualization or complex graph queries.

This order keeps the MVP grounded: concept search proves the graph is useful to users,
workspace branching proves multi-user isolation, and Neo4j then becomes a real
visualization/query backend rather than decorative infrastructure.

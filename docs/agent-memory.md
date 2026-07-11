# Agent Memory Design

This project separates agent memory from the paper knowledge graph.

The paper knowledge graph should model paper content only: papers, sections, chunks,
figures, tables, equations, concepts and eventually claims extracted from the papers
themselves. User preferences, conversation history, generated answers and runtime task
state belong to agent memory or artifacts, not to the paper graph.

## Three-layer agent memory

### 1. Short-term session memory

Purpose: keep the current run coherent.

Examples:

- current workspace ID;
- current paper ID;
- last query;
- last answer artifact path;
- active task.

Implementation:

- `paper_agent.memory.SessionState`
- `paper_agent.memory.SessionMemoryStore`
- default path: `artifacts/memory/session.json`

### 2. Episodic memory

Purpose: remember what happened without treating it as paper knowledge.

Examples:

- a user added `LLaVA.pdf`;
- an answer was generated and saved as an artifact;
- a retrieval evaluation was run;
- the user clarified that paper graph must not store user profile data.

Implementation:

- `paper_agent.memory.Episode`
- `paper_agent.memory.EpisodicMemoryStore`
- default path: `artifacts/memory/episodes.jsonl`

### 3. Long-term user profile memory

Purpose: store stable user and environment preferences.

Examples:

- preferred language;
- default workspace ID;
- server project path;
- Conda environment name;
- preferred command style.

Implementation:

- `paper_agent.memory.UserProfile`
- `paper_agent.memory.UserProfileStore`
- default path: `artifacts/memory/user_profile.json`

## Artifact memory

Answer bundles, evaluation outputs and parsed paper artifacts are reproducible project
artifacts. They are not graph nodes by default.

Examples:

- `artifacts/answers/*.answer.json`
- `artifacts/evals/*.json`
- `artifacts/papers/{paper_id}/paper.json`
- `artifacts/papers/{paper_id}/chunks.json`

Useful command:

```bash
python scripts/list_answers.py --answers artifacts/answers
```

## Promotion policy

An answer can be considered for long-term memory only if it is reusable and evidence
grounded. Even then, the policy should ask the user instead of silently persisting it.

Current policy:

- abstained answers are rejected;
- answers with no claims are rejected;
- answers whose semantic judge report is not fully supported are rejected;
- otherwise the agent may ask the user whether to archive or save it.

This policy is implemented in `paper_agent.memory.PromotionPolicy`.

## Relationship to the paper graph

```text
Agent Memory            Paper Knowledge Graph
-------------           ---------------------
session state           Paper
episodes                Section
user profile            Chunk
answer artifacts        Figure / Table / Equation
evaluation artifacts    Concept
                        PaperClaim (future)
```

The graph can be used by the agent as an external knowledge store, but it is not the
agent's memory system.

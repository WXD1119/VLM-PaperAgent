# Web Reading Desk

The first web version is a local/remote-development reading desk rather than a
public multi-user deployment. It keeps the heavy models on the Linux server and
forwards only the Streamlit web port to the browser.

## Responsibilities

| Layer | Responsibility |
|---|---|
| FastAPI (`:8000`) | Runs retrieval, reranking, answer generation, citation validation, context guard and read-only Paper KG queries. |
| GLM judge (`:8765`, optional) | Independently checks whether each generated claim is supported by its cited evidence. |
| Streamlit (`:8501`) | Presents the reading desk: question input, answer/evidence view, context decision and concept explorer. |

The Streamlit UI never loads BGE, reranker, Qwen or GLM itself. This avoids
duplicating GPU memory and makes it safe to use through VS Code port forwarding.

## API surface

- `GET /health`: service liveness check.
- `POST /ask`: evidence-grounded paper QA.
- `GET /papers`: papers visible in the configured Paper KG/workspace.
- `GET /graph/concepts?q=...`: concept candidates and their related papers,
  chunks, sections and concepts. Ambiguous concepts are returned as candidates
  rather than silently selecting one.
- `POST /ingestion/tasks`: upload a PDF and create a durable ingestion task.
- `GET /ingestion/tasks`: list recent task state, including the current stage and
  useful worker logs.
- `POST /ingestion/tasks/{task_id}/retry`: return a failed task to the queue.

## Run on the server

Install the optional packages once:

```bash
pip install -e '.[api,ui]'
```

In terminal A, start the optional independent judge when semantic gating is
needed:

```bash
conda activate judge-vlm
cd /workspace/guest/wxd/VLM-PaperAgent
CUDA_VISIBLE_DEVICES=2,3 python scripts/serve_glm_judge.py \
  --model artifacts/models/GLM-4.1V-9B-Thinking \
  --host 127.0.0.1 --port 8765 --max-memory-gib 12
```

In terminal B, start the application API. Adjust GPU numbers to the currently
available devices.

```bash
conda activate paper-agent
cd /workspace/guest/wxd/VLM-PaperAgent

export PAPER_AGENT_CHUNKS=artifacts/papers
export PAPER_AGENT_CHROMA_DB=artifacts/chroma
export PAPER_AGENT_EMBEDDING_MODEL=artifacts/models/bge-m3
export PAPER_AGENT_RERANKER_MODEL=artifacts/models/bge-reranker-v2-m3
export PAPER_AGENT_LLM_MODEL=artifacts/models/Qwen3-VL-8B-Instruct
export PAPER_AGENT_OFFLINE=1
export PAPER_AGENT_DEVICE=cuda:0
export PAPER_AGENT_LLM_DEVICE=cuda:1
export PAPER_AGENT_GRAPH=artifacts/graph
export PAPER_AGENT_GRAPH_WORKSPACE=artifacts/graph_workspaces/ws_wxd_demo
export PAPER_AGENT_JUDGE_URL=http://127.0.0.1:8765

uvicorn paper_agent.api.main:app --host 127.0.0.1 --port 8000
```

In terminal C, start the web UI:

```bash
conda activate paper-agent
cd /workspace/guest/wxd/VLM-PaperAgent
PAPER_AGENT_API_URL=http://127.0.0.1:8000 \
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Forward port `8501` in VS Code's **Ports** panel, then open the forwarded URL in
your local browser. `8000` and `8765` remain server-internal: Streamlit calls
the API, and the API calls the judge.

## Upload worker

The web page creates a task but deliberately does not run OCR inside the HTTP
request. In terminal D, run the sequential worker after uploading a PDF:

```bash
conda activate paper-agent
cd /workspace/guest/wxd/VLM-PaperAgent

# If MinerU only exists in its own conda environment, use this command prefix.
export PAPER_AGENT_MINERU_COMMAND='conda run -n mineru mineru'

python scripts/run_ingestion_worker.py --once
```

The task transitions through `pending → running → succeeded` (or `failed`) and
records stages `mineru`, `normalize`, `chunk`, `index`, and
`ready_for_promotion`. It updates the retrieval corpus only after the PDF has
produced valid paper and chunk artifacts. The Paper KG is deliberately not
changed yet: click **提交到个人图谱分支** in the web page to create a workspace
commit. Use `--once` for a controlled single upload; omit it to keep a
long-running worker polling for new tasks.

## Smoke tests

With API running:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/papers
curl 'http://127.0.0.1:8000/graph/concepts?q=Q-Former'
python -m pytest tests/test_api.py -v
python -m pytest tests/test_ingestion_jobs.py -v
```

Expected: the health endpoint returns `{"status":"ok"}`, `/papers` lists the
visible papers, and a concept response contains either one resolved concept or
explicit candidates requiring disambiguation.

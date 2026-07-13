# M8: VLM Benchmark v2 Dataset Preparation

M8 expands the project from a small seed/dev set toward a larger VLM-focused
benchmark. The goal is not to claim a final benchmark yet; it is to create a
reproducible data pipeline for collecting papers, drafting questions and then
human-verifying evidence chunks.

## Paper Set

The current VLM benchmark manifest contains 10 representative papers:

| # | Short name | arXiv | Theme |
|---:|---|---|---|
| 1 | CLIP | 2103.00020 | contrastive pretraining |
| 2 | ALIGN | 2102.05918 | noisy text supervision |
| 3 | ALBEF | 2107.07651 | align-before-fuse |
| 4 | BLIP | 2201.12086 | bootstrapped pretraining |
| 5 | Flamingo | 2204.14198 | few-shot visual language model |
| 6 | CoCa | 2205.01917 | contrastive captioning |
| 7 | BLIP-2 | 2301.12597 | frozen encoder/LLM bridge |
| 8 | LLaVA | 2304.08485 | visual instruction tuning |
| 9 | MiniGPT-4 | 2304.10592 | frozen VLM projection |
| 10 | InstructBLIP | 2305.06500 | instruction-tuned VLM |

Manifest:

```text
evals/vlm_benchmark_manifest.v2.json
```

Downloaded PDFs:

```text
data/raw/vlm_benchmark/*.pdf
```

## Candidate Questions

The manifest includes 5 candidate question seeds per paper, producing 50 draft
questions. The questions intentionally cover multiple categories:

- method;
- dataset;
- evaluation;
- comparison;
- limitation.

They also cover multiple expected evidence kinds:

- text;
- table;
- figure;
- equation.

Generate the draft question queue:

```bash
python scripts/build_vlm_benchmark_questions.py \
  --manifest evals/vlm_benchmark_manifest.v2.json \
  --output evals/retrieval_questions.v2.draft.json
```

## Download

```bash
python scripts/download_vlm_papers.py \
  --manifest evals/vlm_benchmark_manifest.v2.json \
  --output-dir data/raw/vlm_benchmark
```

The downloader skips existing files by default, so it is safe to rerun after
temporary network failures.

## Batch Preparation

Inspect current preparation status:

```bash
python scripts/inspect_vlm_benchmark.py
```

Dry-run the batch preparation plan and print the MinerU command:

```bash
python scripts/prepare_vlm_benchmark.py \
  --print-mineru-command \
  --dry-run
```

Run MinerU on the 10-paper PDF directory. On the current server, `pipeline` is the
safer backend when the hybrid/VLM engine hits CUDA-driver compatibility issues:

```bash
CUDA_VISIBLE_DEVICES=2 \
MINERU_MODEL_SOURCE=modelscope \
mineru \
  -p data/raw/vlm_benchmark \
  -o artifacts/mineru_vlm_benchmark \
  -b pipeline \
  -m auto
```

After MinerU finishes, convert all available MinerU content lists into `paper.json`
and `chunks.json`:

```bash
python scripts/prepare_vlm_benchmark.py \
  --mineru-output artifacts/mineru_vlm_benchmark \
  --papers-output artifacts/papers
```

The script is resumable: papers without MinerU output are reported as missing and
skipped; papers with output are parsed and chunked.

After all target papers are chunked, run post-processing. The default mode is a dry
run that prints the dense-index, graph, annotation and evaluation commands:

```bash
python scripts/postprocess_vlm_benchmark.py
```

Execute dense indexing and graph building:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/postprocess_vlm_benchmark.py \
  --execute \
  --device cuda:0
```

This uses a separate Chroma collection/database for the v2 benchmark by default:

```text
artifacts/chroma_vlm_benchmark
paper_chunks_bge_m3_vlm_v2
```

Build a `paper_key -> paper_id` map before annotation. This resolves draft questions
such as `paper_key=clip` into the actual hash-based `paper_id`, so the annotator does
not have to type paper IDs manually:

```bash
python scripts/build_vlm_paper_id_map.py \
  --manifest evals/vlm_benchmark_manifest.v2.json \
  --papers artifacts/papers \
  --output evals/vlm_paper_id_map.v2.json
```

## Next Annotation Workflow

The draft questions are not yet a golden set. For retrieval evaluation, each item
must be human-verified against parsed chunks:

1. Parse PDFs with MinerU.
2. Convert MinerU outputs into `paper.json`.
3. Build `chunks.json`.
4. Rebuild dense index and graph artifacts.
5. Run interactive retrieval annotation against the 50 draft questions.
6. Save verified cases as `evals/retrieval_golden.v2.json`.

Recommended annotation command after parsing/chunking:

```bash
python scripts/annotate_retrieval.py \
  --chunks artifacts/papers \
  --questions evals/retrieval_questions.v2.draft.json \
  --paper-id-map evals/vlm_paper_id_map.v2.json \
  --output evals/retrieval_golden.v2.json \
  --annotator wxd \
  --top-k 12
```

If `--paper-id-map` is present, the annotator automatically restricts each query to
the intended paper. If it is missing, the script falls back to prompting for
`paper_id (optional)`.

Only `retrieval_golden.v2.json` should be used for final retrieval metrics.
The draft file is a question queue, not ground truth.

## Retrieval Metrics

The v2 benchmark uses multi-evidence annotations: one question can have several
relevant chunks because a good paper-reading answer may need method text, table
results and ablation evidence together.

For that reason, the evaluation reports both binary hit metrics and evidence-coverage
metrics:

- `Hit@K`: whether at least one relevant evidence chunk appears in the top K results.
  This is the most intuitive "can the answer agent start answering?" metric.
- `Recall@K`: how many of all annotated relevant chunks appear in the top K results.
  This is stricter and measures multi-evidence coverage.
- `MRR`: reciprocal rank of the first relevant evidence chunk.
- `nDCG@5`: ranking quality for multiple relevant chunks in the first five results.

Low `Recall@1` is expected for broad comparison questions with many relevant chunks.
Use `Hit@1`, `MRR`, `Recall@5` and `nDCG@5` together when explaining retrieval quality.

## Dense Index Guardrail

Dense, hybrid and reranked evaluation depend on the Chroma vector index. Before
running final metrics, check that the vector collection is not empty and roughly
matches the parsed chunk count. The evaluation scripts now print `collection_count`
and fail fast when the dense collection is empty.

If `collection_count` is zero, rebuild the dense index:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/index_dense.py \
  --chunks artifacts/papers \
  --db artifacts/chroma \
  --model artifacts/models/bge-m3 \
  --offline \
  --device cuda:0
```

If `collection_count` differs from the loaded chunk count, treat the result as a
stale-index warning and rebuild the index unless the difference is intentionally
explained.

## Current v2 Retrieval Results

The current v2 run uses 48 manually annotated retrieval cases over 10 VLM papers.

| Variant | Hit@1 | Hit@5 | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6250 | 1.0000 | 0.1679 | 0.6144 | 0.7733 | 0.6042 |
| BGE-M3 dense | 0.7083 | 1.0000 | 0.2030 | 0.6005 | 0.8420 | 0.6377 |
| BM25 + BGE-M3 RRF | 0.7917 | 1.0000 | 0.2252 | 0.6606 | 0.8889 | 0.6906 |
| RRF + BGE reranker v2 m3 | 0.7500 | 1.0000 | 0.2099 | 0.6944 | 0.8611 | 0.6943 |

中文结论：

- 四种方法的 `Hit@5` 都是 1.0，说明前 5 条结果总能找到至少一个可用证据。
- `BM25 + BGE-M3 RRF` 的 `Hit@1` 和 `MRR` 最好，适合作为默认在线召回方案。
- `RRF + BGE reranker` 的 `Recall@5` 和 `nDCG@5` 最好，适合需要更强多证据覆盖的高质量回答模式。
- `Recall@1` 看起来偏低，是因为 v2 是多证据标注；一个问题经常需要多个 chunk 才能完整回答，top-1 很难覆盖完整证据集合。

工程复盘：

- 首次 v2 dense/hybrid 评测发现 `collection_count=0`，导致 Dense 检索异常偏低。
- 重新执行 `index_dense.py` 后结果恢复正常。
- 后续新增 `index_dense.py --reset`，用于删除并重建 Chroma collection，避免旧索引残留。

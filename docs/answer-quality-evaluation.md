# M6: Answer Quality and Hallucination-Risk Evaluation

M6 turns "hallucination" into measurable answer-level metrics.

This project already has evidence-grounded generation and semantic citation judging.
M6 aggregates those artifacts into experiment metrics.

## Inputs

```text
artifacts/answers/*.answer.json
artifacts/evals/*.glm-judge.json
```

Answer bundles provide:

- citation integrity;
- abstention status;
- claim count;
- evidence pack.

GLM judge reports provide semantic support labels:

- `supported`;
- `partially_supported`;
- `unsupported`.

## Metrics

- `Citation pass rate`: fraction of answers whose citations are structurally valid.
- `Abstention rate`: fraction of answers that abstained.
- `Semantic coverage rate`: fraction of generated claims assessed by the semantic judge.
- `Semantic support rate`: supported claims / assessed claims.
- `Unsupported claim rate`: partially supported or unsupported claims / assessed claims.
- `Fully supported answer rate`: answers whose citations pass and all claims are
  semantically supported.

`Unsupported claim rate` is the main hallucination-risk proxy.

## Command

```bash
python scripts/evaluate_answer_quality.py \
  --answers artifacts/answers \
  --judges artifacts/evals \
  --output artifacts/evals/answer_quality.json
```

Use quality gates when the evaluation should behave like a CI check:

```bash
python scripts/evaluate_answer_quality.py \
  --answers artifacts/answers \
  --judges artifacts/evals \
  --output artifacts/evals/answer_quality.json \
  --min-semantic-coverage 0.90 \
  --min-semantic-support 0.95 \
  --max-unsupported-claim-rate 0.05 \
  --min-fully-supported-answer-rate 0.70 \
  --fail-on-gate
```

The script prints `Quality gate: PASS` or `Quality gate: FAIL`. On failure it also
lists the failed metrics and risky answer cases. The JSON report is still written
before exiting, so failed runs remain inspectable.

## Increasing semantic coverage

If `Semantic coverage rate` is low, batch-judge missing answers:

```bash
python scripts/batch_judge_answers.py \
  --answers artifacts/answers \
  --output-dir artifacts/evals \
  --judge-url http://127.0.0.1:8765 \
  --dry-run

python scripts/batch_judge_answers.py \
  --answers artifacts/answers \
  --output-dir artifacts/evals \
  --judge-url http://127.0.0.1:8765 \
  --keep-going
```

Default behavior:

- skip answers that already have `*.glm-judge.json`;
- skip abstained answers because they have no factual claims;
- judge only answers with claims;
- use the already-running judge service, avoiding repeated model reloads.

## Relation to RAGAS

RAGAS can be added later as an optional external evaluator, especially for context
precision/recall and answer relevance. The current M6 evaluator is kept dependency-free
and aligned with this project's citation-first design.

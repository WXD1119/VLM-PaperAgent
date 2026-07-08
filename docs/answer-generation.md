# Evidence-grounded answer generation

## Model routing decision

- Default: local `Qwen3-VL-8B-Instruct` over the retrieved `EvidencePack`, with no per-token fee.
- Escalation: a stronger reasoning model when the first pass abstains or citation validation fails.
- Vision fallback: a multimodal GLM-family model only when a selected figure requires pixels and
  its caption/OCR does not answer the question.

This keeps the common path inexpensive while preserving a genuine multimodal route. MinerU has
already converted most PDF content into text, table HTML, LaTeX, and captions, so invoking a VLM
for every question would add cost without adding visual information.

## Safety contract

1. Retrieval results receive stable request-local IDs (`E1`, `E2`, ...).
2. The model returns structured claims with one or more evidence IDs.
3. The validator rejects citations outside the evidence pack.
4. Insufficient evidence produces an explicit abstention rather than an unsupported answer.

Citation integrity and semantic support are reported separately. The optional semantic judge
classifies every claim as `supported`, `partially_supported`, or `unsupported` using only its cited
evidence. Because the generator and judge currently share one local checkpoint, this is a useful
runtime guard rather than an unbiased evaluation metric; manually labeled citation evaluation
remains necessary.

The local client uses Transformers directly instead of vLLM because the current server has an
older NVIDIA 535 driver. This avoids introducing another CUDA runtime compatibility constraint.

For independent evaluation, GLM runs in a separate Conda environment behind a localhost-only
HTTP service. `RemoteStructuredClient` lets the answer process call it without sharing Python,
Transformers, or CUDA dependencies. The service must not bind to a public interface because it has
no authentication layer.

On limited hardware, prefer sequential offline judging: `answer_question.py --output` persists the
answer and exact evidence pack, then exits and releases Qwen. `judge_answer.py` loads that immutable
bundle in the isolated GLM environment. This avoids concurrent model residency and CPU offload.

Human citation evaluation uses saved answer bundles rather than regenerated text. Annotators label
each claim as supported, partially supported, or unsupported while viewing only its cited evidence.
The judge is scored with claim accuracy, macro F1, supported precision/recall, and abstention
accuracy. Model verdicts are never copied into the human Golden Set.

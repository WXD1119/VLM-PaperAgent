# Golden Set

## Context Guard

`context_guard.seed.json` is a small deterministic golden set for memory-aware context
guard behavior. It checks whether ambiguous follow-up questions are constrained to the
current session paper, whether missing context triggers clarification, and whether clear
new questions avoid stale-session constraints.

Run:

```bash
python scripts/evaluate_context_guard.py \
  --golden evals/context_guard.seed.json \
  --output artifacts/evals/context_guard.seed.json
```

建议从 10 篇 CV/VLM 论文开始，标注 80-120 个问题及证据元素 ID。禁止提交受版权限制的论文原文；这里只保存可公开的元数据、问题、答案摘要和证据定位。

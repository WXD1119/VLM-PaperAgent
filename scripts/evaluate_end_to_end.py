"""调用运行中的 API，对人工标注检索集执行端到端证据问答评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from paper_agent.evaluation import EndToEndResponse, RetrievalCase, evaluate_end_to_end


def _ask(base_url: str, case: RetrievalCase, top_k: int) -> EndToEndResponse:
    payload = json.dumps(
        {
            "query": case.query,
            "paper_id": case.paper_id,
            "kind": case.kind.value if case.kind else None,
            "top_k": top_k,
            "use_context_guard": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/ask",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        raw = json.loads(response.read().decode("utf-8"))
    bundle = raw["bundle"]
    return EndToEndResponse(
        evidence_chunk_ids=[item["chunk_id"] for item in bundle["evidence_pack"]["items"]],
        citation_valid=bundle["citation_validation"]["valid"],
        abstained=bundle["answer"]["abstained"],
        semantic_gate_enabled=raw.get("semantic_gate_enabled", False),
        semantic_gate_passed=raw.get("semantic_gate_passed"),
        attempts=raw.get("attempts", 1),
        rewrite_count=raw.get("rewrite_count", 0),
        refusal_kind=raw.get("refusal_kind"),
        trace_id=raw.get("trace_id"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评测运行中 API 的端到端证据问答闭环")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [RetrievalCase.model_validate(item) for item in json.loads(args.golden.read_text(encoding="utf-8"))]
    if args.limit is not None:
        cases = cases[: args.limit]
    responses = {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.query_id}")
        responses[case.query_id] = _ask(args.api_url, case, args.top_k)
    evaluation = evaluate_end_to_end(cases, responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    print(f"cases: {evaluation.case_count}")
    print(f"Evidence Hit@5: {evaluation.evidence_hit_at_5:.4f}")
    print(f"Evidence Recall@5: {evaluation.evidence_recall_at_5:.4f}")
    print(f"Citation pass rate: {evaluation.citation_pass_rate:.4f}")
    print(f"Semantic gate coverage: {evaluation.semantic_gate_coverage:.4f}")
    print(f"Semantic gate pass rate: {evaluation.semantic_gate_pass_rate:.4f}")
    print(f"Rewrite rate: {evaluation.rewrite_rate:.4f}")
    print(f"Safe refusal rate: {evaluation.safe_refusal_rate:.4f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

from paper_agent.evaluation import EndToEndResponse, RetrievalCase, evaluate_end_to_end


def test_end_to_end_evaluation_reports_evidence_and_quality_loop_metrics():
    cases = [
        RetrievalCase(query_id="supported", query="q1", relevant_chunk_ids={"c1"}),
        RetrievalCase(query_id="refused", query="q2", relevant_chunk_ids={"c2"}),
    ]
    result = evaluate_end_to_end(
        cases,
        {
            "supported": EndToEndResponse(
                evidence_chunk_ids=["c1"],
                citation_valid=True,
                abstained=False,
                semantic_gate_enabled=True,
                semantic_gate_passed=True,
                attempts=1,
            ),
            "refused": EndToEndResponse(
                evidence_chunk_ids=["c2"],
                citation_valid=True,
                abstained=True,
                semantic_gate_enabled=True,
                semantic_gate_passed=False,
                attempts=2,
                rewrite_count=1,
                refusal_kind="semantic_support_failed",
            ),
        },
    )
    assert result.evidence_hit_at_5 == 1.0
    assert result.citation_pass_rate == 1.0
    assert result.semantic_gate_coverage == 1.0
    assert result.rewrite_rate == 0.5
    assert result.safe_refusal_rate == 1.0

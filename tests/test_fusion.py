from paper_agent.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_cross_list_hits() -> None:
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert result[0][0] == "b"


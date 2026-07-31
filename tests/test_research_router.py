from paper_agent.orchestration import AgentName, CorpusScope, ResearchRouteRequest, ResearchRouter


def test_router_dispatches_active_explanation_then_library_discovery():
    decision = ResearchRouter().route(
        ResearchRouteRequest(
            query="先帮我理解这篇论文，再看看库里是否有人提过这个想法",
            active_paper_id="paper-current",
        )
    )
    assert [item.mode for item in decision.dispatches] == ["active_paper_explain", "library_related_work"]
    assert decision.dispatches[0].corpus_scope == CorpusScope.ACTIVE_PAPER_ONLY
    assert decision.dispatches[1].depends_on == ["active_paper_explain"]


def test_router_requires_two_selected_papers_for_compare():
    decision = ResearchRouter().route(ResearchRouteRequest(query="比较这两篇论文的方法"))
    assert decision.dispatches[0].agent == AgentName.READING_COMPARE
    assert decision.needs_clarification is True


def test_explicit_review_requires_confirmation_without_selected_papers():
    decision = ResearchRouter().route(ResearchRouteRequest(query="写综述", mode="review"))
    assert decision.dispatches[0].agent == AgentName.REVIEW_WRITER
    assert decision.dispatches[0].needs_confirmation is True

from paper_agent.domain import ChunkKind
from paper_agent.evaluation.routing import RoutingGoldenCase, evaluate_routing


def test_routing_evaluation_scores_expected_router_outputs():
    report = evaluate_routing(
        [
            RoutingGoldenCase(
                case_id="formula",
                query="解释式(3)的训练目标",
                intent="formula_explanation",
                paper_scope=["paper-flow"],
                expected_evidence_types=[ChunkKind.EQUATION, ChunkKind.TEXT],
                expected_use_graph=False,
                expected_needs_clarification=False,
            ),
            RoutingGoldenCase(
                case_id="ambiguous-table",
                query="表2中哪个方法更好？",
                intent="follow_up",
                expected_evidence_types=[ChunkKind.TEXT],
                expected_use_graph=False,
                expected_needs_clarification=True,
            ),
        ]
    )

    assert report.case_count == 2
    assert report.intent_accuracy == 1.0
    assert report.clarification_accuracy == 1.0

from dataclasses import dataclass

from paper_agent.domain import AnswerBundle
from paper_agent.graph.fragments import build_answer_fragment, delta_from_fragment
from paper_agent.graph.validation import GraphValidationReport, GraphValidator
from paper_agent.graph.workspace import GraphCommit, GraphDelta, LocalGraphWorkspaceStore


@dataclass(frozen=True)
class AnswerPromotionResult:
    """Result of promoting an immutable answer artifact into a graph workspace."""

    query: str
    delta: GraphDelta
    validation: GraphValidationReport
    commit: GraphCommit | None = None

    @property
    def changed(self) -> bool:
        return bool(self.delta.added_nodes or self.delta.added_edges)


def promote_answer_to_workspace(
    answer: AnswerBundle,
    store: LocalGraphWorkspaceStore,
    *,
    author_id: str,
    message: str | None = None,
    allow_invalid: bool = False,
    dry_run: bool = False,
) -> AnswerPromotionResult:
    """Promote a saved answer bundle into a user's long-term graph workspace.

    Answer generation itself should remain transient: `ask.py` writes an immutable
    answer artifact, and this function is called only when a user or workflow decides
    that the answer is valuable enough to become persistent graph memory.
    """

    fragment = build_answer_fragment(answer)
    delta = delta_from_fragment(store.load_effective_graph(), fragment)
    preview = store.preview_delta(delta)
    report = GraphValidator().validate(preview)
    if not report.valid and not allow_invalid:
        return AnswerPromotionResult(
            query=answer.evidence_pack.query,
            delta=delta,
            validation=report,
            commit=None,
        )
    if dry_run or not delta.added_nodes and not delta.added_edges:
        return AnswerPromotionResult(
            query=answer.evidence_pack.query,
            delta=delta,
            validation=report,
            commit=None,
        )

    commit = store.commit_delta(
        delta,
        author_id=author_id,
        message=message or f"promote answer for query: {answer.evidence_pack.query}",
    )
    return AnswerPromotionResult(
        query=answer.evidence_pack.query,
        delta=delta,
        validation=report,
        commit=commit,
    )

import argparse

from paper_agent.memory import (
    ContextGuard,
    SessionMemoryStore,
    UserProfileStore,
    load_graph_entity_resolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain how session/profile memory constrains an ambiguous query"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--paper-id", default=None, help="Explicit paper_id from the user.")
    parser.add_argument("--session", default="artifacts/memory/session.json")
    parser.add_argument("--profile", default="artifacts/memory/user_profile.json")
    parser.add_argument("--graph", default="artifacts/graph")
    parser.add_argument("--graph-workspace", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolver = load_graph_entity_resolver(
        graph_path=args.graph,
        workspace_path=args.graph_workspace,
    )
    decision = ContextGuard(resolver).decide(
        args.query,
        explicit_paper_id=args.paper_id,
        session=SessionMemoryStore(args.session).load(),
        profile=UserProfileStore(args.profile).load(),
    )
    print(f"action: {decision.action.value}")
    print(f"query: {decision.query}")
    print(f"resolved_paper_id: {decision.resolved_paper_id or 'none'}")
    print(f"needs_clarification: {decision.needs_clarification}")
    if decision.clarification_question:
        print(f"clarification_question: {decision.clarification_question}")
    if decision.candidate_paper_ids:
        print("candidate_paper_ids:")
        for paper_id in decision.candidate_paper_ids:
            print(f"- {paper_id}")
    if decision.warnings:
        print("warnings:")
        for warning in decision.warnings:
            print(f"- {warning}")
    if decision.context_notes:
        print("context_notes:")
        for note in decision.context_notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()

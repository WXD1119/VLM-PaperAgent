import argparse
from pathlib import Path

from paper_agent.domain import ChunkBundle, Paper
from paper_agent.graph import (
    GraphValidator,
    LocalGraphWorkspaceStore,
    build_paper_fragment,
    delta_from_fragment,
)
from paper_agent.memory import EpisodicMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a parsed paper to a graph workspace")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--paper", required=True, help="Path to paper.json")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--author", required=True)
    parser.add_argument("--message", default=None)
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--memory", default="artifacts/memory/episodes.jsonl")
    args = parser.parse_args()

    paper = Paper.model_validate_json(Path(args.paper).read_text(encoding="utf-8"))
    chunks = ChunkBundle.model_validate_json(Path(args.chunks).read_text(encoding="utf-8"))
    fragment = build_paper_fragment(paper, chunks)
    store = LocalGraphWorkspaceStore(args.workspace)
    delta = delta_from_fragment(store.load_effective_graph(), fragment)

    if not delta.added_nodes and not delta.added_edges:
        print(f"no new graph records for paper: {paper.paper_id}")
        return

    preview = store.preview_delta(delta)
    report = GraphValidator().validate(preview)
    if not report.valid and not args.allow_invalid:
        print("paper delta would make effective graph invalid:")
        for error in report.errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    commit = store.commit_delta(
        delta,
        author_id=args.author,
        message=args.message or f"add paper {paper.paper_id}",
    )
    episode = EpisodicMemoryStore(args.memory).log(
        "paper_added_to_workspace",
        f"Added paper {paper.paper_id} to workspace {args.workspace}",
        {
            "workspace": args.workspace,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "commit_id": commit.commit_id,
            "added_nodes": len(delta.added_nodes),
            "added_edges": len(delta.added_edges),
        },
    )
    print(f"paper_id: {paper.paper_id}")
    print(f"title: {paper.title}")
    print(f"commit_id: {commit.commit_id}")
    print(f"valid_after_commit: {report.valid}")
    print(f"added_nodes: {len(delta.added_nodes)}")
    print(f"added_edges: {len(delta.added_edges)}")
    print(f"memory_event_id: {episode.event_id}")


if __name__ == "__main__":
    main()

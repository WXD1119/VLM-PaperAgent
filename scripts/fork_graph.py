import argparse

from paper_agent.graph import LocalGraphWorkspaceStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Fork a base graph into a graph workspace")
    parser.add_argument(
        "--base",
        required=True,
        help="Base graph directory containing nodes/edges JSONL",
    )
    parser.add_argument("--output", required=True, help="Workspace output directory")
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--message", default="fork workspace")
    args = parser.parse_args()

    workspace = LocalGraphWorkspaceStore(args.output).create_fork(
        base_graph_path=args.base,
        workspace_id=args.workspace_id,
        owner_id=args.owner,
        name=args.name,
        message=args.message,
    )
    print(f"workspace_id: {workspace.workspace_id}")
    print(f"owner_id: {workspace.owner_id}")
    print(f"name: {workspace.name}")
    print(f"base_graph_path: {workspace.base_graph_path}")
    print(f"head_commit_id: {workspace.head_commit_id}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

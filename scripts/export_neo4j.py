import argparse
import os
from pathlib import Path

from paper_agent.graph import LocalGraphWorkspaceStore, read_graph_jsonl
from paper_agent.storage import Neo4jGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a JSONL Paper KG or workspace effective graph into Neo4j"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", type=Path, help="Base JSONL graph directory")
    source.add_argument("--workspace", type=Path, help="Graph workspace directory")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--reset", action="store_true", help="Delete existing materialized nodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.getenv(args.password_env)
    if not password:
        raise SystemExit(f"environment variable {args.password_env} is required")
    graph = (
        LocalGraphWorkspaceStore(args.workspace).load_effective_graph()
        if args.workspace
        else read_graph_jsonl(args.graph)
    )
    with Neo4jGraphStore(args.uri, args.user, password, database=args.database) as store:
        report = store.replace_graph(graph, reset=args.reset)
        nodes, edges = store.counts()
    print(f"source: {args.workspace or args.graph}")
    print(f"written_nodes: {report.node_count}")
    print(f"written_edges: {report.edge_count}")
    print(f"database_nodes: {nodes}")
    print(f"database_edges: {edges}")
    print(f"reset: {report.reset}")


if __name__ == "__main__":
    main()

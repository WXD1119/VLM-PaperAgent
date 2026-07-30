import argparse
import os

from paper_agent.storage import Neo4jGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the Neo4j materialized Paper KG")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--papers", action="store_true")
    action.add_argument("--concept", help="Exact concept label, for example Q-Former")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password-env", default="NEO4J_PASSWORD")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.getenv(args.password_env)
    if not password:
        raise SystemExit(f"environment variable {args.password_env} is required")
    with Neo4jGraphStore(args.uri, args.user, password, database=args.database) as store:
        rows = store.papers() if args.papers else store.concept_papers(args.concept, args.limit)
    if not rows:
        print("No results.")
        return
    for row in rows:
        print(f"{row['paper_id']}  {row['title']}")


if __name__ == "__main__":
    main()

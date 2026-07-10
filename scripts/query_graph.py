import argparse

from paper_agent.graph import GraphQuery, NodeType, read_graph_jsonl


def parse_node_types(values: list[str] | None) -> list[NodeType] | None:
    if not values:
        return None
    return [NodeType(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the lightweight evidence graph")
    parser.add_argument("--graph", default="artifacts/graph")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--papers", action="store_true", help="List paper nodes")
    parser.add_argument("--paper-id", default=None, help="Paper ID for paper-scoped queries")
    parser.add_argument("--chunks", action="store_true", help="List chunks for --paper-id")
    parser.add_argument("--claim-supports", action="store_true", help="List claim -> evidence chunks")
    parser.add_argument("--search", default=None, help="Substring search over graph nodes")
    parser.add_argument(
        "--node-type",
        action="append",
        choices=[node_type.value for node_type in NodeType],
        help="Restrict --search to one or more node types",
    )
    args = parser.parse_args()

    if not any([args.papers, args.chunks, args.claim_supports, args.search]):
        parser.error("choose one query: --papers, --chunks, --claim-supports, or --search")
    if args.chunks and not args.paper_id:
        parser.error("--chunks requires --paper-id")

    query = GraphQuery(read_graph_jsonl(args.graph))
    if args.papers:
        for node in query.papers()[: args.limit]:
            print(f"{node.properties.get('paper_id')}\t{node.label}")

    if args.chunks:
        for node in query.chunks_for_paper(args.paper_id, args.limit):
            pages = node.properties.get("pages", [])
            section = " > ".join(node.properties.get("section_path", []))
            print(
                f"{node.properties.get('chunk_id')}\t{node.properties.get('kind')}"
                f"\tpages={pages}\t{section}"
            )

    if args.claim_supports:
        for claim, chunks in query.claim_supports(args.limit):
            print(f"CLAIM {claim.properties.get('claim_index')}: {claim.properties.get('text')}")
            for chunk in chunks:
                pages = chunk.properties.get("pages", [])
                section = " > ".join(chunk.properties.get("section_path", []))
                print(
                    f"  -> {chunk.properties.get('chunk_id')} "
                    f"paper={chunk.properties.get('paper_id')} pages={pages} section={section}"
                )

    if args.search:
        node_types = parse_node_types(args.node_type)
        for node in query.search_nodes(args.search, node_types=node_types, limit=args.limit):
            print(f"{node.node_type.value}\t{node.node_id}\t{node.label}")


if __name__ == "__main__":
    main()

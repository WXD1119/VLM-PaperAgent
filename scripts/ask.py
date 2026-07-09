import argparse
import os
from pathlib import Path

from paper_agent.agents import AnswerAgent, build_evidence_pack
from paper_agent.domain import AnswerBundle, ChunkKind
from paper_agent.llm import OpenAICompatibleClient, TransformersStructuredClient
from paper_agent.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerEncoder,
    chunks_from_bundles,
    load_chunk_bundles,
)
from paper_agent.storage import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end paper QA demo: hybrid retrieval, optional reranking, "
            "evidence-grounded answer generation, and citation integrity validation."
        )
    )
    parser.add_argument("--query", required=True, help="Question to ask over the paper corpus.")
    parser.add_argument(
        "--chunks",
        default="artifacts/papers",
        help="Directory containing chunks.json files.",
    )
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--embedding-model", default="artifacts/models/bge-m3")
    parser.add_argument("--reranker-model", default="artifacts/models/bge-reranker-v2-m3")
    parser.add_argument(
        "--model",
        default=os.getenv("PAPER_AGENT_LLM_MODEL", "artifacts/models/Qwen3-VL-8B-Instruct"),
        help="Local HF model path or OpenAI-compatible model name.",
    )
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument(
        "--base-url",
        default=os.getenv("PAPER_AGENT_LLM_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--device", default=None, help="Embedding/reranker device, e.g. cuda:0.")
    parser.add_argument("--llm-device", default=None, help="Local generator device, e.g. cuda:1.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load local retrieval models without network access.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Use RRF results directly without cross-encoder reranking.",
    )
    parser.add_argument("--paper-id", default=None, help="Optional paper_id filter.")
    parser.add_argument("--kind", choices=[kind.value for kind in ChunkKind], default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/answers/ask.answer.json"))
    parser.add_argument(
        "--content-chars",
        type=int,
        default=700,
        help="Max evidence content chars to print.",
    )
    return parser.parse_args()


def build_demo_retriever(args: argparse.Namespace):
    chunks = chunks_from_bundles(load_chunk_bundles(args.chunks))
    sparse = BM25Index(chunks)
    encoder = SentenceTransformerEncoder(
        args.embedding_model, device=args.device, local_files_only=args.offline
    )
    dense = ChromaVectorStore(args.db, args.collection, encoder)
    hybrid = HybridRetriever(sparse, dense, args.rrf_k, args.candidate_k)
    if args.no_rerank:
        return hybrid
    reranker = CrossEncoderReranker(
        args.reranker_model, device=args.device, local_files_only=args.offline
    )
    return RerankedRetriever(hybrid, reranker, args.candidate_k)


def build_llm_client(args: argparse.Namespace):
    if args.provider == "local":
        return TransformersStructuredClient(args.model, device=args.llm_device)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is required")
    return OpenAICompatibleClient(args.model, api_key, args.base_url)


def clip_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if limit < 1 or len(compact) <= limit:
        return compact
    if limit <= 3:
        return "." * limit
    return compact[: max(0, limit - 3)].rstrip() + "..."


def render_bundle(bundle: AnswerBundle, *, content_chars: int = 700) -> str:
    answer = bundle.answer
    validation = bundle.citation_validation
    lines = [
        "# Answer",
        "",
        answer.answer,
        "",
        f"Abstained: {answer.abstained}",
    ]
    if answer.abstention_reason:
        lines.append(f"Abstention reason: {answer.abstention_reason}")

    lines.extend(["", "# Claims"])
    if answer.claims:
        for index, claim in enumerate(answer.claims, start=1):
            evidence = " ".join(f"[{evidence_id}]" for evidence_id in claim.evidence_ids)
            lines.append(f"{index}. {claim.text} {evidence}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "# Citation integrity",
            "",
            f"Status: {'PASS' if validation.valid else 'FAIL'}",
            f"Claims: {validation.claim_count}",
            f"Cited claims: {validation.cited_claim_count}",
        ]
    )
    if validation.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in validation.errors)

    lines.extend(["", "# Evidence"])
    for item in bundle.evidence_pack.items:
        section = " > ".join(item.section_path) or "(unknown section)"
        lines.extend(
            [
                "",
                f"[{item.evidence_id}] {item.paper_id} | {item.kind.value} | pages={item.pages}",
                f"chunk_id={item.chunk_id}",
                f"section={section}",
            ]
        )
        if item.image_path:
            lines.append(f"image_path={item.image_path}")
        lines.append(clip_text(item.content, content_chars))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    retriever = build_demo_retriever(args)
    kind = ChunkKind(args.kind) if args.kind else None
    hits = retriever.search(args.query, args.top_k, paper_id=args.paper_id, kind=kind)
    pack = build_evidence_pack(args.query, hits)
    answer, validation = AnswerAgent(build_llm_client(args)).answer(pack)
    bundle = AnswerBundle(
        evidence_pack=pack,
        answer=answer,
        citation_validation=validation,
        generator_model=args.model,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    print(render_bundle(bundle, content_chars=args.content_chars))
    print("\n# Run")
    print(f"retriever={'RRF' if args.no_rerank else 'RRF+BGE-reranker'}")
    print(f"embedding_model={args.embedding_model}")
    if not args.no_rerank:
        print(f"reranker_model={args.reranker_model}")
    print(f"generator_model={args.model}")
    print(f"answer_bundle={args.output}")


if __name__ == "__main__":
    main()

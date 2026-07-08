import argparse
import os
from pathlib import Path

from paper_agent.agents import AnswerAgent, SemanticCitationJudge, build_evidence_pack
from paper_agent.llm import (
    OpenAICompatibleClient,
    RemoteStructuredClient,
    TransformersStructuredClient,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a paper question with traceable evidence")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/chroma"))
    parser.add_argument("--collection", default="paper_chunks_bge_m3")
    parser.add_argument("--embedding-model", default="artifacts/models/bge-m3")
    parser.add_argument("--reranker-model", default="artifacts/models/bge-reranker-v2-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-device", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--provider", choices=("local", "openai"), default="local")
    parser.add_argument(
        "--model",
        default=os.getenv("PAPER_AGENT_LLM_MODEL", "artifacts/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("PAPER_AGENT_LLM_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--verify-semantics", action="store_true")
    parser.add_argument("--judge-url", default=None)
    args = parser.parse_args()

    chunks = chunks_from_bundles(load_chunk_bundles(args.chunks))
    sparse = BM25Index(chunks)
    encoder = SentenceTransformerEncoder(
        args.embedding_model, device=args.device, local_files_only=args.offline
    )
    dense = ChromaVectorStore(args.db, args.collection, encoder)
    hybrid = HybridRetriever(sparse, dense, args.rrf_k, args.candidate_k)
    reranker = CrossEncoderReranker(
        args.reranker_model, device=args.device, local_files_only=args.offline
    )
    retriever = RerankedRetriever(hybrid, reranker, args.candidate_k)
    hits = retriever.search(args.query, args.top_k)
    pack = build_evidence_pack(args.query, hits)

    if args.provider == "local":
        client = TransformersStructuredClient(args.model, device=args.llm_device)
    else:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            parser.error(f"environment variable {args.api_key_env} is required")
        client = OpenAICompatibleClient(args.model, api_key, args.base_url)
    answer, validation = AnswerAgent(client).answer(pack)
    print(answer.answer)
    print(f"\nAbstained: {answer.abstained}")
    if answer.abstention_reason:
        print(f"Abstention reason: {answer.abstention_reason}")
    print("\nClaims and citations:")
    for claim in answer.claims:
        print(f"- {claim.text} [{' '.join(claim.evidence_ids)}]")
    print(f"\nCitation validation: {'PASS' if validation.valid else 'FAIL'}")
    for item in pack.items:
        print(
            f"[{item.evidence_id}] {item.paper_id} pages={item.pages} "
            f"chunk_id={item.chunk_id}"
        )
    if args.verify_semantics:
        judge_client = RemoteStructuredClient(args.judge_url) if args.judge_url else client
        semantic = SemanticCitationJudge(judge_client).evaluate(answer, pack)
        print("\nSemantic citation validation:")
        if answer.abstained:
            print("SKIPPED (answer abstained)")
        else:
            for item in semantic.assessments:
                print(
                    f"- claim={item.claim_index} verdict={item.verdict.value} "
                    f"evidence={item.evidence_ids}: {item.reasoning_summary}"
                )
            print(f"All claims supported: {semantic.all_supported}")


if __name__ == "__main__":
    main()

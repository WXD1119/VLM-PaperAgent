from paper_agent.domain.chunk import ChunkKind, RetrievalChunk
from paper_agent.retrieval.sparse import BM25Index, tokenize_academic


def chunk(
    chunk_id: str,
    content: str,
    kind: ChunkKind = ChunkKind.TEXT,
    paper_id: str = "paper_x",
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        parent_chunk_id="parent_x",
        paper_id=paper_id,
        kind=kind,
        content=content,
        element_ids=[f"element_{chunk_id}"],
        pages=[1],
        section_path=["Method"],
    )


def test_academic_tokenizer_removes_html_and_preserves_latex_commands() -> None:
    tokens = tokenize_academic("<table><tr><td>Flow</td></tr></table> $\\theta_1$")
    assert "table" not in tokens
    assert "td" not in tokens
    assert "flow" in tokens
    assert "\\theta" in tokens


def test_bm25_ranks_rare_exact_term_first() -> None:
    index = BM25Index(
        [
            chunk("generic", "vision language model training"),
            chunk("qformer", "Q-Former bootstraps frozen image encoders"),
            chunk("other", "contrastive representation learning"),
        ]
    )
    hits = index.search("Q-Former", top_k=2)
    assert hits[0].chunk_id == "qformer"
    assert hits[0].score > 0


def test_bm25_searches_latex_and_filters_kind_and_paper() -> None:
    index = BM25Index(
        [
            chunk("formula", r"$$v_\\theta(x_t,t)=u_t(x_t)$$", ChunkKind.EQUATION),
            chunk("prose", "theta is a parameter", ChunkKind.TEXT),
            chunk("other-paper", r"$$v_\\theta$$", ChunkKind.EQUATION, "paper_y"),
        ]
    )
    hits = index.search(
        r"\\theta",
        top_k=5,
        paper_id="paper_x",
        kind=ChunkKind.EQUATION,
    )
    assert [hit.chunk_id for hit in hits] == ["formula"]


def test_bm25_empty_or_unknown_query_returns_no_hits() -> None:
    index = BM25Index([chunk("known", "known vocabulary")])
    assert index.search("") == []
    assert index.search("unseen") == []

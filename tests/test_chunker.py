from paper_agent.domain.paper import ElementType, Paper, PaperElement
from paper_agent.ingestion.chunker import PaperChunker


def element(
    element_id: str,
    content: str,
    element_type: ElementType = ElementType.PARAGRAPH,
    section: list[str] | None = None,
    **metadata,
) -> PaperElement:
    return PaperElement(
        element_id=element_id,
        paper_id="paper_x",
        page=1,
        section_path=section or ["Title", "Method"],
        element_type=element_type,
        content=content,
        parser_version="test",
        metadata=metadata,
    )


def paper_with_multimodal_elements() -> Paper:
    return Paper(
        paper_id="paper_x",
        title="Title",
        source_path="paper.pdf",
        sha256="a" * 64,
        elements=[
            element("h", "Method", text_level=2),
            element("p1", "Previous explanation with $x_t$."),
            element("eq", "$$x_t=x_0+t(x_1-x_0)$$", ElementType.EQUATION),
            element("p2", "Following explanation."),
            element("t", "Table 1\n<table/>", ElementType.TABLE),
            element("f", "Figure 1: Architecture", ElementType.FIGURE),
            element("foot", "conference footer", retrieval_excluded=True),
        ],
    )


def test_chunker_preserves_special_elements_and_equation_context() -> None:
    bundle = PaperChunker().chunk(paper_with_multimodal_elements())
    by_kind = {chunk.kind.value: chunk for chunk in bundle.children if chunk.kind.value != "text"}

    assert set(by_kind) == {"equation", "table", "figure"}
    assert "Previous explanation" in by_kind["equation"].context
    assert "Following explanation" in by_kind["equation"].context
    assert by_kind["equation"].element_ids == ["eq"]
    assert all("foot" not in chunk.element_ids for chunk in bundle.children)


def test_chunker_marks_inline_latex_and_links_to_parent() -> None:
    bundle = PaperChunker().chunk(paper_with_multimodal_elements())
    text_chunks = [chunk for chunk in bundle.children if chunk.kind.value == "text"]

    assert text_chunks[0].metadata["contains_inline_latex"] is True
    parent_ids = {parent.parent_chunk_id for parent in bundle.parents}
    assert all(chunk.parent_chunk_id in parent_ids for chunk in bundle.children)
    assert "Section: Title > Method" in text_chunks[0].embedding_text


def test_chunk_ids_are_stable() -> None:
    chunker = PaperChunker()
    first = chunker.chunk(paper_with_multimodal_elements())
    second = chunker.chunk(paper_with_multimodal_elements())
    assert [chunk.chunk_id for chunk in first.children] == [
        chunk.chunk_id for chunk in second.children
    ]


def test_chunker_splits_long_text_with_unique_ids() -> None:
    long_text = " ".join(f"Sentence {index}." for index in range(100))
    paper = Paper(
        paper_id="paper_x",
        title="Title",
        source_path="paper.pdf",
        sha256="a" * 64,
        elements=[element("long", long_text)],
    )

    chunks = PaperChunker(max_chars=220).chunk(paper).children

    assert len(chunks) > 1
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(len(chunk.content) <= 220 for chunk in chunks)
    assert all(chunk.metadata["part_count"] == len(chunks) for chunk in chunks)


def test_chunker_splits_html_table_and_repeats_caption_and_header() -> None:
    rows = "".join(f"<tr><td>{index}</td><td>{'x' * 70}</td></tr>" for index in range(12))
    table = "Table 1: Scores\n\n<table><tr><th>ID</th><th>Value</th></tr>" + rows + "</table>"
    paper = Paper(
        paper_id="paper_x",
        title="Title",
        source_path="paper.pdf",
        sha256="a" * 64,
        elements=[element("table", table, ElementType.TABLE)],
    )

    chunks = PaperChunker(max_chars=300).chunk(paper).children

    assert len(chunks) > 1
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.content.startswith("Table 1: Scores") for chunk in chunks)
    assert all("<th>ID</th>" in chunk.content for chunk in chunks)
    assert all(chunk.content.endswith("</table>") for chunk in chunks)

import json

import pytest

from paper_agent.domain.paper import ElementType
from paper_agent.ingestion.mineru_adapter import MinerUAdapter, paper_id_from_pdf


def make_fixture(tmp_path):
    pdf = tmp_path / "BLIP2.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    content_list = tmp_path / "BLIP2_content_list.json"
    content_list.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "BLIP-2: Bootstrapping",
                    "text_level": 1,
                    "bbox": [10, 20, 100, 40],
                    "page_idx": 0,
                },
                {
                    "type": "text",
                    "text": "Abstract",
                    "text_level": 2,
                    "bbox": [10, 50, 60, 70],
                    "page_idx": 0,
                },
                {
                    "type": "text",
                    "text": "  Frozen   encoders.  ",
                    "bbox": [10, 80, 300, 120],
                    "page_idx": 0,
                },
                {
                    "type": "interline_equation",
                    "latex": "E = mc^2",
                    "bbox": [20, 20, 200, 60],
                    "page_idx": 1,
                },
                {
                    "type": "image",
                    "img_path": "images/figure.jpg",
                    "image_caption": ["Figure 1: Architecture"],
                    "bbox": [20, 70, 400, 300],
                    "page_idx": 1,
                },
                {
                    "type": "header",
                    "text": "Conference header",
                    "bbox": [0, 0, 100, 10],
                    "page_idx": 1,
                },
                {
                    "type": "chart",
                    "img_path": "images/chart.jpg",
                    "image_caption": ["Figure 2: Accuracy"],
                    "bbox": [20, 310, 400, 500],
                    "page_idx": 1,
                },
                {
                    "type": "table",
                    "table_body": "<table><tr><td>76.2</td></tr></table>",
                    "table_caption": ["Table 1: ImageNet accuracy"],
                    "bbox": [20, 510, 400, 700],
                    "page_idx": 1,
                },
            ]
        ),
        encoding="utf-8",
    )
    return pdf, content_list


def test_adapter_maps_pages_sections_types_and_links(tmp_path) -> None:
    pdf, content_list = make_fixture(tmp_path)
    paper_id = paper_id_from_pdf(pdf)

    paper = MinerUAdapter(content_list).parse(str(pdf), paper_id)

    assert paper.title == "BLIP-2: Bootstrapping"
    assert len(paper.elements) == 7
    assert paper.elements[2].content == "Frozen encoders."
    assert paper.elements[2].section_path == ["BLIP-2: Bootstrapping", "Abstract"]
    assert paper.elements[2].parent_id == paper.elements[1].element_id
    assert paper.elements[3].page == 2
    assert paper.elements[3].element_type == ElementType.EQUATION
    assert paper.elements[4].element_type == ElementType.FIGURE
    assert paper.elements[4].content == "Figure 1: Architecture"
    assert paper.elements[5].element_type == ElementType.FIGURE
    assert paper.elements[5].content == "Figure 2: Accuracy"
    assert paper.elements[6].element_type == ElementType.TABLE
    assert paper.elements[6].content.startswith("Table 1: ImageNet accuracy")
    assert "<table>" in paper.elements[6].content
    assert all(item.content != "Conference header" for item in paper.elements)
    assert paper.elements[0].next_id == paper.elements[1].element_id
    assert paper.elements[1].prev_id == paper.elements[0].element_id


def test_adapter_produces_stable_ids(tmp_path) -> None:
    pdf, content_list = make_fixture(tmp_path)
    adapter = MinerUAdapter(content_list)
    first = adapter.parse(str(pdf), "paper_x")
    second = adapter.parse(str(pdf), "paper_x")
    assert [item.element_id for item in first.elements] == [
        item.element_id for item in second.elements
    ]


def test_adapter_rejects_invalid_page_index(tmp_path) -> None:
    pdf, content_list = make_fixture(tmp_path)
    content_list.write_text(
        json.dumps([{"type": "text", "text": "x", "page_idx": -1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="page_idx"):
        MinerUAdapter(content_list).parse(str(pdf), "paper_x")

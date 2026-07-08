import hashlib
import json
from pathlib import Path
from typing import Any

from paper_agent.domain.paper import ElementType, Paper, PaperElement
from paper_agent.ingestion.element_linker import link_adjacent_elements
from paper_agent.ingestion.normalizer import normalize_text


_TYPE_MAP = {
    "text": ElementType.PARAGRAPH,
    "aside_text": ElementType.PARAGRAPH,
    "page_footnote": ElementType.PARAGRAPH,
    "equation": ElementType.EQUATION,
    "interline_equation": ElementType.EQUATION,
    "inline_equation": ElementType.EQUATION,
    "table": ElementType.TABLE,
    "image": ElementType.FIGURE,
    "figure": ElementType.FIGURE,
    "chart": ElementType.FIGURE,
    "image_caption": ElementType.CAPTION,
    "figure_caption": ElementType.CAPTION,
    "table_caption": ElementType.CAPTION,
}

_SKIP_TYPES = {"header", "page_number"}


class MinerUAdapter:
    """Convert MinerU 3.x content_list.json into stable domain models."""

    def __init__(self, content_list_path: str | Path, version: str = "mineru-3.4.2") -> None:
        self.content_list_path = Path(content_list_path)
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def parse(self, pdf_path: str, paper_id: str) -> Paper:
        source = Path(pdf_path)
        if not source.is_file():
            raise FileNotFoundError(f"PDF does not exist: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"expected a PDF file: {source}")
        if not self.content_list_path.is_file():
            raise FileNotFoundError(f"MinerU content list does not exist: {self.content_list_path}")

        raw = json.loads(self.content_list_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("MinerU content_list.json must contain a JSON array")

        elements: list[PaperElement] = []
        headings: dict[int, tuple[str, str]] = {}
        for source_index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type", "text"))
            if raw_type in _SKIP_TYPES:
                continue
            content = self._extract_content(item)
            if not content:
                continue

            element_type = _TYPE_MAP.get(raw_type, ElementType.PARAGRAPH)
            page_idx = self._page_index(item)
            bbox = self._bbox(item.get("bbox"))
            heading_level = self._heading_level(item)

            section_path = [headings[level][0] for level in sorted(headings)]
            parent_id = headings[max(headings)][1] if headings else None
            element_id = self._element_id(
                paper_id, source_index, page_idx, raw_type, content
            )

            if heading_level is not None:
                headings = {level: value for level, value in headings.items() if level < heading_level}
                section_path = [headings[level][0] for level in sorted(headings)] + [content]
                parent_id = headings[max(headings)][1] if headings else None
                headings[heading_level] = (content, element_id)

            content_fields = {
                "text", "content", "latex", "table_body", "bbox", "page_idx"
            }
            metadata = {
                key: value
                for key, value in item.items()
                if key not in content_fields
            }
            metadata["mineru_type"] = raw_type
            metadata["source_index"] = source_index
            if raw_type == "page_footnote":
                metadata["retrieval_excluded"] = True
            if raw_type not in _TYPE_MAP:
                metadata["unmapped_type"] = True

            elements.append(
                PaperElement(
                    element_id=element_id,
                    paper_id=paper_id,
                    page=page_idx + 1,
                    bbox=bbox,
                    section_path=section_path,
                    element_type=element_type,
                    content=content,
                    parent_id=parent_id,
                    parser_version=self.version,
                    metadata=metadata,
                )
            )

        if not elements:
            raise ValueError("MinerU content list did not contain any usable elements")

        title = self._find_title(elements, source.stem)
        return Paper(
            paper_id=paper_id,
            title=title,
            source_path=str(source),
            sha256=_sha256_file(source),
            elements=link_adjacent_elements(elements),
        )

    @staticmethod
    def _extract_content(item: dict[str, Any]) -> str:
        raw_type = str(item.get("type", "text"))
        if raw_type == "table":
            return MinerUAdapter._captioned_content(
                item, ("table_caption", "caption"), ("table_body", "text", "content")
            )
        if raw_type == "chart":
            return MinerUAdapter._captioned_content(
                item,
                ("chart_caption", "image_caption", "caption"),
                ("chart_body", "text", "content"),
            ) or MinerUAdapter._image_fallback(item)

        candidates = (
            item.get("text"),
            item.get("latex"),
            item.get("table_body"),
            item.get("content"),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and normalize_text(candidate):
                return normalize_text(candidate)

        for key in ("image_caption", "table_caption", "caption"):
            caption = item.get(key)
            if isinstance(caption, str) and normalize_text(caption):
                return normalize_text(caption)
            if isinstance(caption, list):
                text = normalize_text("\n".join(str(value) for value in caption if value))
                if text:
                    return text

        return MinerUAdapter._image_fallback(item)

    @staticmethod
    def _captioned_content(
        item: dict[str, Any], caption_keys: tuple[str, ...], body_keys: tuple[str, ...]
    ) -> str:
        parts: list[str] = []
        for key in caption_keys:
            value = item.get(key)
            if isinstance(value, str):
                parts.append(value)
                break
            if isinstance(value, list):
                parts.extend(str(part) for part in value if part)
                break
        for key in body_keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
                break
        return normalize_text("\n\n".join(parts))

    @staticmethod
    def _image_fallback(item: dict[str, Any]) -> str:
        image_path = item.get("img_path") or item.get("image_path")
        if isinstance(image_path, str) and image_path.strip():
            return f"Image: {image_path.strip()}"
        return ""

    @staticmethod
    def _page_index(item: dict[str, Any]) -> int:
        value = item.get("page_idx", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid MinerU page_idx: {value!r}")
        return value

    @staticmethod
    def _bbox(value: Any) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError(f"invalid MinerU bbox: {value!r}")
        try:
            bbox = tuple(float(number) for number in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid MinerU bbox: {value!r}") from exc
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(f"invalid MinerU bbox ordering: {value!r}")
        return bbox  # type: ignore[return-value]

    @staticmethod
    def _heading_level(item: dict[str, Any]) -> int | None:
        level = item.get("text_level")
        if isinstance(level, bool) or not isinstance(level, int) or level < 1:
            return None
        return level

    @staticmethod
    def _element_id(
        paper_id: str, source_index: int, page_idx: int, raw_type: str, content: str
    ) -> str:
        identity = f"{paper_id}\0{source_index}\0{page_idx}\0{raw_type}\0{content}"
        return f"elem_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _find_title(elements: list[PaperElement], fallback: str) -> str:
        for element in elements:
            if element.metadata.get("text_level") == 1:
                return element.content
        return elements[0].content if elements else fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paper_id_from_pdf(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF does not exist: {source}")
    return f"paper_{_sha256_file(source)[:16]}"

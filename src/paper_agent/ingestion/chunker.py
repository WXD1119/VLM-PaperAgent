import hashlib
import re
from collections import defaultdict

from paper_agent.domain.chunk import ChunkBundle, ChunkKind, ParentChunk, RetrievalChunk
from paper_agent.domain.paper import ElementType, Paper, PaperElement


class PaperChunker:
    """Create section parents and evidence-preserving retrieval children."""

    def __init__(self, max_chars: int = 2800, overlap_elements: int = 1) -> None:
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")
        if overlap_elements < 0:
            raise ValueError("overlap_elements cannot be negative")
        self.max_chars = max_chars
        self.overlap_elements = overlap_elements

    def chunk(self, paper: Paper) -> ChunkBundle:
        eligible = [
            element
            for element in paper.elements
            if not element.metadata.get("retrieval_excluded")
        ]
        content_elements = [element for element in eligible if not self._is_heading(element)]
        parents = self._build_parents(paper.paper_id, content_elements)
        parent_by_section = {
            tuple(parent.section_path): parent.parent_chunk_id for parent in parents
        }
        children = self._build_children(paper.paper_id, eligible, parent_by_section)
        return ChunkBundle(paper_id=paper.paper_id, parents=parents, children=children)

    def _build_parents(
        self, paper_id: str, elements: list[PaperElement]
    ) -> list[ParentChunk]:
        grouped: dict[tuple[str, ...], list[PaperElement]] = defaultdict(list)
        for element in elements:
            grouped[tuple(element.section_path)].append(element)

        parents: list[ParentChunk] = []
        for section_path, section_elements in grouped.items():
            element_ids = [element.element_id for element in section_elements]
            parents.append(
                ParentChunk(
                    parent_chunk_id=_stable_id(
                        "parent", paper_id, "\0".join(section_path) or "root"
                    ),
                    paper_id=paper_id,
                    section_path=list(section_path),
                    content="\n\n".join(element.content for element in section_elements),
                    element_ids=element_ids,
                    pages=sorted({element.page for element in section_elements}),
                )
            )
        return parents

    def _build_children(
        self,
        paper_id: str,
        elements: list[PaperElement],
        parent_by_section: dict[tuple[str, ...], str],
    ) -> list[RetrievalChunk]:
        children: list[RetrievalChunk] = []
        text_buffer: list[PaperElement] = []

        def flush_text() -> None:
            nonlocal text_buffer
            if not text_buffer:
                return
            children.append(
                self._make_chunk(
                    paper_id,
                    ChunkKind.TEXT,
                    text_buffer,
                    parent_by_section,
                )
            )
            text_buffer = []

        for index, element in enumerate(elements):
            if self._is_heading(element):
                flush_text()
                continue
            if element.element_type == ElementType.PARAGRAPH:
                if len(element.content) > self.max_chars:
                    flush_text()
                    parts = _split_long_text(element.content, self.max_chars)
                    for part_index, part in enumerate(parts):
                        children.append(
                            self._make_chunk(
                                paper_id,
                                ChunkKind.TEXT,
                                [element],
                                parent_by_section,
                                content_override=part,
                                split_info=(part_index, len(parts)),
                            )
                        )
                    continue
                if text_buffer and (
                    text_buffer[-1].section_path != element.section_path
                    or self._buffer_size(text_buffer, element) > self.max_chars
                ):
                    overlap = text_buffer[-self.overlap_elements :] if self.overlap_elements else []
                    flush_text()
                    text_buffer = (
                        overlap
                        if self._buffer_size(overlap, element) <= self.max_chars
                        else []
                    )
                text_buffer.append(element)
                continue

            flush_text()
            kind = {
                ElementType.EQUATION: ChunkKind.EQUATION,
                ElementType.TABLE: ChunkKind.TABLE,
                ElementType.FIGURE: ChunkKind.FIGURE,
                ElementType.CAPTION: ChunkKind.FIGURE,
            }[element.element_type]
            context = ""
            if kind == ChunkKind.EQUATION:
                context = self._equation_context(elements, index, element.section_path)
            parts = (
                _split_html_table(element.content, self.max_chars)
                if kind == ChunkKind.TABLE
                else [element.content]
            )
            for part_index, part in enumerate(parts):
                children.append(
                    self._make_chunk(
                        paper_id,
                        kind,
                        [element],
                        parent_by_section,
                        context=context,
                        content_override=part,
                        split_info=(part_index, len(parts)),
                    )
                )
        flush_text()
        return children

    def _make_chunk(
        self,
        paper_id: str,
        kind: ChunkKind,
        elements: list[PaperElement],
        parent_by_section: dict[tuple[str, ...], str],
        context: str = "",
        content_override: str | None = None,
        split_info: tuple[int, int] | None = None,
    ) -> RetrievalChunk:
        section_path = elements[0].section_path
        element_ids = [element.element_id for element in elements]
        content = content_override or "\n\n".join(element.content for element in elements)
        parent_id = parent_by_section[tuple(section_path)]
        content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        part_index, part_count = split_info or (0, 1)
        return RetrievalChunk(
            chunk_id=_stable_id(
                "child", paper_id, kind.value, *element_ids, content_digest
            ),
            parent_chunk_id=parent_id,
            paper_id=paper_id,
            kind=kind,
            content=content,
            context=context,
            element_ids=element_ids,
            pages=sorted({element.page for element in elements}),
            section_path=section_path,
            metadata={
                "char_count": len(content),
                "contains_inline_latex": kind == ChunkKind.TEXT and _contains_latex(content),
                "oversized": len(content) > self.max_chars,
                "part_index": part_index,
                "part_count": part_count,
            },
        )

    @staticmethod
    def _equation_context(
        elements: list[PaperElement], index: int, section_path: list[str]
    ) -> str:
        neighbors: list[str] = []
        for direction in (-1, 1):
            position = index + direction
            while 0 <= position < len(elements):
                candidate = elements[position]
                if candidate.section_path != section_path:
                    break
                is_context = (
                    candidate.element_type == ElementType.PARAGRAPH
                    and not PaperChunker._is_heading(candidate)
                )
                if is_context:
                    if not candidate.metadata.get("retrieval_excluded"):
                        neighbors.append(candidate.content)
                    break
                position += direction
        return "\n\n".join(neighbors)

    @staticmethod
    def _is_heading(element: PaperElement) -> bool:
        level = element.metadata.get("text_level")
        return isinstance(level, int) and not isinstance(level, bool) and level >= 1

    @staticmethod
    def _buffer_size(buffer: list[PaperElement], candidate: PaperElement) -> int:
        return sum(len(element.content) + 2 for element in buffer) + len(candidate.content)


def _contains_latex(content: str) -> bool:
    return "$" in content or "\\begin{" in content or "\\[" in content


def _split_long_text(content: str, max_chars: int) -> list[str]:
    """Split prose near sentence boundaries while avoiding open $...$ spans."""
    parts: list[str] = []
    remaining = content.strip()
    while len(remaining) > max_chars:
        lower_bound = max_chars // 2
        candidates: list[int] = []
        for pattern in (r"\n\n", r"(?<=[.!?])\s+", r"(?<=;)\s+"):
            candidates.extend(
                match.end()
                for match in re.finditer(pattern, remaining[: max_chars + 1])
                if match.end() >= lower_bound and _outside_inline_math(remaining, match.end())
            )
        split_at = max(candidates, default=max_chars)
        part = remaining[:split_at].strip()
        if not part:
            split_at = max_chars
            part = remaining[:split_at]
        parts.append(part)
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _outside_inline_math(content: str, position: int) -> bool:
    prefix = content[:position]
    dollars = len(re.findall(r"(?<!\\)\$", prefix))
    return dollars % 2 == 0


def _split_html_table(content: str, max_chars: int) -> list[str]:
    """Split MinerU HTML tables by rows and repeat caption plus first header row."""
    table_start = re.search(r"<table(?:\s[^>]*)?>", content, flags=re.IGNORECASE)
    if not table_start:
        return _split_long_text(content, max_chars)
    table_end = re.search(r"</table>\s*$", content, flags=re.IGNORECASE)
    if not table_end:
        return _split_long_text(content, max_chars)

    prefix = content[: table_start.start()].strip()
    open_tag = table_start.group(0)
    table_body = content[table_start.end() : table_end.start()]
    rows = re.findall(r"<tr(?:\s[^>]*)?>.*?</tr>", table_body, flags=re.I | re.S)
    if len(rows) < 2:
        return [content]

    header = rows[0]
    groups: list[list[str]] = []
    current: list[str] = []
    for row in rows[1:]:
        candidate = _render_table_part(prefix, open_tag, header, current + [row])
        if current and len(candidate) > max_chars:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)

    parts = [_render_table_part(prefix, open_tag, header, group) for group in groups]
    return parts if parts and all(parts) else [content]


def _render_table_part(
    prefix: str, open_tag: str, header: str, rows: list[str]
) -> str:
    table = f"{open_tag}{header}{''.join(rows)}</table>"
    return f"{prefix}\n\n{table}" if prefix else table


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"

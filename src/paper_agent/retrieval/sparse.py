import math
import re
from collections import Counter
from html import unescape
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from paper_agent.domain.chunk import ChunkBundle, ChunkKind, RetrievalChunk


_HTML_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(
    r"\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]"
)


def tokenize_academic(text: str) -> list[str]:
    """为词法检索切分正文、表格 HTML 与 LaTeX 命令。"""
    clean = unescape(_HTML_TAG.sub(" ", text))
    return [token.lower() for token in _TOKEN.findall(clean)]


class SparseHit(BaseModel):
    chunk_id: str
    paper_id: str
    kind: ChunkKind
    score: float = Field(ge=0)
    pages: list[int]
    section_path: list[str]
    content: str
    context: str = ""


class BM25Index:
    """轻量、无额外依赖且可复现的 BM25Okapi 基线索引。"""

    def __init__(
        self,
        chunks: Iterable[RetrievalChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)
        self._tokens = [tokenize_academic(chunk.embedding_text) for chunk in self.chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequencies = self._compute_document_frequencies(self._tokens)
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        kind: ChunkKind | None = None,
    ) -> list[SparseHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize_academic(query)
        if not query_tokens or not self.chunks:
            return []

        query_frequency = Counter(query_tokens)
        scored: list[tuple[float, RetrievalChunk]] = []
        for index, chunk in enumerate(self.chunks):
            if paper_id is not None and chunk.paper_id != paper_id:
                continue
            if kind is not None and chunk.kind != kind:
                continue
            score = self._score_document(index, query_frequency)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SparseHit(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                kind=chunk.kind,
                score=score,
                pages=chunk.pages,
                section_path=chunk.section_path,
                content=chunk.content,
                context=chunk.context,
            )
            for score, chunk in scored[:top_k]
        ]

    def _score_document(self, index: int, query_frequency: Counter[str]) -> float:
        frequencies = self._term_frequencies[index]
        document_length = len(self._tokens[index])
        score = 0.0
        for token, query_count in query_frequency.items():
            term_frequency = frequencies.get(token, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequencies.get(token, 0)
            inverse_document_frequency = math.log(
                1
                + (len(self.chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = term_frequency + self.k1 * (
                1 - self.b
                + self.b * document_length / max(self._average_length, 1.0)
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (self.k1 + 1)
                / normalization
                * query_count
            )
        return score

    @staticmethod
    def _compute_document_frequencies(
        tokenized_documents: list[list[str]],
    ) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in tokenized_documents:
            frequencies.update(set(tokens))
        return frequencies


def load_chunk_bundles(root: str | Path) -> list[ChunkBundle]:
    root_path = Path(root)
    paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("chunks.json"))
    if not paths:
        raise FileNotFoundError(f"no chunks.json found below {root_path}")
    return [ChunkBundle.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]


def chunks_from_bundles(bundles: Iterable[ChunkBundle]) -> list[RetrievalChunk]:
    return [chunk for bundle in bundles for chunk in bundle.children]

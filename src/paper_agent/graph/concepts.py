import re
from dataclasses import dataclass


_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "their",
    "this",
    "through",
    "using",
    "with",
}

_TECH_KEYWORDS = {
    "alignment",
    "attention",
    "classification",
    "contrastive",
    "decoder",
    "diffusion",
    "embedding",
    "encoder",
    "flow",
    "generation",
    "image",
    "language",
    "learning",
    "matching",
    "model",
    "pre-training",
    "prompt",
    "query",
    "representation",
    "text",
    "transformer",
    "vector",
    "vision",
    "zero-shot",
}

_KNOWN_PHRASES = [
    "q-former",
    "frozen image encoder",
    "frozen language model",
    "frozen llm",
    "learnable query embeddings",
    "query embeddings",
    "cross-attention",
    "soft visual prompts",
    "conditional flow matching",
    "flow matching",
    "vector field",
    "interpolation path",
    "zero-shot",
    "linear probe",
    "image encoder",
    "language model",
]

_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)+\b|\b[A-Z]{2,}(?:\d+)?\b")
_TECH_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Za-z]+(?:-[A-Za-z0-9]+)+|[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,3})\b"
)


@dataclass(frozen=True, order=True)
class ConceptMention:
    """A normalized concept mention extracted from graph text."""

    normalized: str
    label: str
    source: str


def extract_concepts(text: str, *, source: str = "rule") -> list[ConceptMention]:
    """Extract deterministic technical concept mentions.

    The first implementation intentionally avoids an LLM dependency. It favors stable,
    explainable technical terms such as `Q-Former`, `CLIP`, `cross-attention`, and
    domain phrases that are useful as graph entry points.
    """

    mentions: dict[str, ConceptMention] = {}
    compact = " ".join(text.split())
    lowered = compact.lower()

    for phrase in _KNOWN_PHRASES:
        if phrase in lowered:
            _add_mention(mentions, phrase, _display_phrase(phrase), source)

    for pattern in (_ACRONYM_PATTERN, _TECH_PHRASE_PATTERN):
        for match in pattern.finditer(compact):
            label = _clean_label(match.group(0))
            if _is_useful_label(label):
                _add_mention(mentions, label, label, source)

    return sorted(mentions.values(), key=lambda item: (item.normalized, item.label))


def normalize_concept(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized


def _add_mention(
    mentions: dict[str, ConceptMention],
    label: str,
    display_label: str,
    source: str,
) -> None:
    normalized = normalize_concept(label)
    if not normalized:
        return
    existing = mentions.get(normalized)
    mention = ConceptMention(normalized=normalized, label=display_label, source=source)
    if existing is None or len(display_label) < len(existing.label):
        mentions[normalized] = mention


def _clean_label(label: str) -> str:
    tokens = label.strip(" \t\n\r.,;:()[]{}<>").split()
    while tokens and tokens[0].lower() in _STOPWORDS:
        tokens.pop(0)
    while tokens and tokens[-1].lower() in _STOPWORDS:
        tokens.pop()
    return " ".join(tokens)


def _is_useful_label(label: str) -> bool:
    if len(label) < 3:
        return False
    normalized = normalize_concept(label)
    if not normalized or normalized in _STOPWORDS:
        return False
    if label.lower() in _STOPWORDS:
        return False
    if _looks_like_person_name(label):
        return False
    words = re.split(r"[\s/-]+", label.lower())
    if " " in label and not any(word in _TECH_KEYWORDS for word in words):
        return False
    return True


def _looks_like_person_name(label: str) -> bool:
    if "-" in label and not any(keyword in label.lower() for keyword in _TECH_KEYWORDS):
        # Keeps technical hyphenated terms such as cross-attention and zero-shot, while
        # filtering citation names such as Ben-Hamu.
        parts = label.split("-")
        return len(parts) == 2 and all(part[:1].isupper() for part in parts if part)
    words = label.split()
    if len(words) == 2 and all(word[:1].isupper() for word in words):
        return not any(word.lower() in _TECH_KEYWORDS for word in words)
    return False


def _display_phrase(phrase: str) -> str:
    special = {
        "frozen llm": "frozen LLM",
        "q-former": "Q-Former",
        "zero-shot": "zero-shot",
    }
    if phrase in special:
        return special[phrase]
    return phrase

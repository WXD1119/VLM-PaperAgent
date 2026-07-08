import re
from html import unescape


_WHITESPACE = re.compile(r"[\t\r\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(value: str) -> str:
    """Normalize parser text without destroying Markdown or LaTeX structure."""
    value = unescape(value).replace("\u00a0", " ")
    lines = [_WHITESPACE.sub(" ", line).strip() for line in value.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


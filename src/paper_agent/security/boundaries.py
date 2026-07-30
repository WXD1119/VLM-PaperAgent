from __future__ import annotations

import re
from dataclasses import dataclass


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_INJECTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous|reveal\s+(?:the\s+)?system\s+prompt|"
    r"you\s+are\s+now|call\s+(?:the\s+)?tool|developer\s+message)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ActorContext:
    """向存储与工具边界传递的已认证主体。"""

    user_id: str
    session_id: str
    scopes: frozenset[str] = frozenset({"paper:read"})

    def __post_init__(self) -> None:
        for label, value in {"user_id": self.user_id, "session_id": self.session_id}.items():
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid {label}")


@dataclass(frozen=True)
class PromptInjectionSignal:
    suspicious: bool
    reasons: tuple[str, ...] = ()


def validate_query(query: str, *, max_chars: int = 4_000) -> PromptInjectionSignal:
    """校验传输安全，同时不阻断正当研究问题。

    用户可能正当询问提示注入；可疑措辞仅作为追踪和更严格工具策略的信号，
    不会触发自动拒答。
    """

    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > max_chars:
        raise ValueError(f"query exceeds {max_chars} characters")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in query):
        raise ValueError("query contains unsupported control characters")
    return PromptInjectionSignal(
        suspicious=bool(_INJECTION.search(query)),
        reasons=("possible_prompt_injection_pattern",) if _INJECTION.search(query) else (),
    )


class ToolAuthorizationPolicy:
    """确定性 Scope 校验；模型不能为自身授予能力。"""

    DEFAULT_SCOPES = {
        "resolve_context": frozenset({"paper:read"}),
        "route_query": frozenset({"paper:read"}),
        "plan_evidence": frozenset({"paper:read"}),
        "retrieve_evidence": frozenset({"paper:read"}),
        "build_evidence": frozenset({"paper:read"}),
        "generate_answer": frozenset({"paper:read"}),
        "validate_citations": frozenset({"paper:read"}),
        "query_paper_graph": frozenset({"paper:read"}),
        "inspect_ingestion_task": frozenset({"paper:read"}),
        "promote_paper_to_workspace": frozenset({"workspace:write"}),
        "semantic_judge": frozenset({"paper:read"}),
    }

    def authorize(self, actor: ActorContext, tool_name: str) -> None:
        required = self.DEFAULT_SCOPES.get(tool_name)
        if required is None:
            raise PermissionError(f"tool is not allowlisted: {tool_name}")
        if not required <= actor.scopes:
            raise PermissionError(f"missing scope for {tool_name}: {', '.join(sorted(required))}")

"""不可信用户问题、文档与工具调用的安全边界。"""

from .boundaries import ActorContext, PromptInjectionSignal, ToolAuthorizationPolicy, validate_query
from .evidence import render_untrusted_evidence
from .identity import CredentialVerifier, TrustedActorResolver, VerifiedIdentity
from .worker_sandbox import NetworkMode, ParserWorkerSandbox, WorkerResourceLimits

__all__ = [
    "ActorContext",
    "CredentialVerifier",
    "NetworkMode",
    "ParserWorkerSandbox",
    "PromptInjectionSignal",
    "ToolAuthorizationPolicy",
    "TrustedActorResolver",
    "VerifiedIdentity",
    "WorkerResourceLimits",
    "render_untrusted_evidence",
    "validate_query",
]

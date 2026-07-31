"""Trusted-identity contracts for the API boundary.

Authentication providers verify credentials outside the agent runtime. This
module deliberately does not decode JWTs itself: accepting unverified claims
would let a request choose its own ``user_id`` or tool scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .boundaries import ActorContext


@dataclass(frozen=True)
class VerifiedIdentity:
    """Claims returned only by a credential-verifying adapter."""

    subject: str
    session_id: str
    scopes: frozenset[str]
    issuer: str
    audience: str
    expires_at: datetime


class CredentialVerifier(Protocol):
    """Adapter implemented by JWT/OIDC/session infrastructure."""

    def verify(self, credential: str) -> VerifiedIdentity: ...


class TrustedActorResolver:
    """Converts verified identity into the narrow actor used by tools."""

    def __init__(self, verifier: CredentialVerifier, *, issuer: str, audience: str) -> None:
        self.verifier = verifier
        self.issuer = issuer
        self.audience = audience

    def resolve(self, credential: str, *, requested_session_id: str | None = None) -> ActorContext:
        if not credential or not credential.strip():
            raise PermissionError("missing credentials")
        identity = self.verifier.verify(credential)
        if identity.issuer != self.issuer or identity.audience != self.audience:
            raise PermissionError("identity issuer or audience is not trusted")
        if identity.expires_at.astimezone(UTC) <= datetime.now(UTC):
            raise PermissionError("identity credentials have expired")
        if requested_session_id is not None and requested_session_id != identity.session_id:
            raise PermissionError("session is not owned by authenticated identity")
        # The agent runtime never grants a default administrative scope.
        return ActorContext(
            user_id=identity.subject,
            session_id=identity.session_id,
            scopes=frozenset(identity.scopes),
        )

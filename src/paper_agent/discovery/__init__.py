"""Confirmed, metadata-only discovery of candidate papers.

Discovery results are deliberately *not* papers in the local corpus.  A user
must review candidates and explicitly import one through the ingestion flow.
"""

from .service import (
    CandidatePaper,
    CandidatePaperDiscovery,
    DiscoveryRequest,
    DiscoveryResult,
    MetadataHttpClient,
)

__all__ = [
    "CandidatePaper",
    "CandidatePaperDiscovery",
    "DiscoveryRequest",
    "DiscoveryResult",
    "MetadataHttpClient",
]

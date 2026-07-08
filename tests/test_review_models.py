import pytest
from pydantic import ValidationError

from paper_agent.domain.review import ReviewFinding, Verdict


def test_supported_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        ReviewFinding(
            claim_id="c1",
            verdict=Verdict.SUPPORTED,
            reasoning_summary="ok",
            confidence=0.9,
        )


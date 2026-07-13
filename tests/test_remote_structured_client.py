import urllib.error
from unittest.mock import patch

import pytest

from paper_agent.llm.client import RemoteStructuredClient


def test_remote_structured_client_accepts_service_root_or_generate_endpoint():
    assert (
        RemoteStructuredClient("http://127.0.0.1:8765").url
        == "http://127.0.0.1:8765/generate"
    )
    assert (
        RemoteStructuredClient("http://127.0.0.1:8765/generate").url
        == "http://127.0.0.1:8765/generate"
    )
    assert (
        RemoteStructuredClient("http://127.0.0.1:8765/generate/").url
        == "http://127.0.0.1:8765/generate"
    )


def test_remote_structured_client_rejects_relative_url():
    with pytest.raises(ValueError, match="scheme and host"):
        RemoteStructuredClient("127.0.0.1:8765")


def test_remote_structured_client_404_error_includes_endpoint_hint():
    client = RemoteStructuredClient("http://127.0.0.1:8765")
    error = urllib.error.HTTPError(
        url=client.url,
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match=r"http://127\.0\.0\.1:8765/generate"):
            client.generate_structured("prompt", _DummyModel)


class _DummyModel:
    @staticmethod
    def model_json_schema():
        return {"title": "Dummy"}

    @staticmethod
    def model_validate(value):
        return value

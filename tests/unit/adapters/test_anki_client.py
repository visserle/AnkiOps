"""Unit tests for the Anki HTTP bridge client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ankiops.anki_client import AnkiConnectError, invoke


def test_invoke_posts_to_ankiops_bridge_and_returns_result():
    response = MagicMock()
    response.json.return_value = {"result": 1, "error": None}

    with patch("ankiops.anki_client._session.post", return_value=response) as post:
        assert invoke("version") == 1

    post.assert_called_once_with(
        "http://127.0.0.1:8766",
        json={"action": "version", "params": {}},
        timeout=10,
    )


def test_invoke_wraps_request_errors_as_anki_connect_error():
    with patch(
        "ankiops.anki_client._session.post",
        side_effect=requests.ConnectionError("Connection reset by peer"),
    ):
        with pytest.raises(AnkiConnectError, match="Unable to reach AnkiOps bridge"):
            invoke("version")


def test_invoke_rejects_non_json_response_payload():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("no json")

    with patch("ankiops.anki_client._session.post", return_value=response):
        with pytest.raises(AnkiConnectError, match="non-JSON response"):
            invoke("version")

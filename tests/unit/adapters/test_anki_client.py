"""Unit tests for the AnkiConnect HTTP client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ankiops.anki_client import AnkiConnectError, invoke


def test_invoke_wraps_request_errors_as_anki_connect_error():
    with patch(
        "ankiops.anki_client._session.post",
        side_effect=requests.ConnectionError("Connection reset by peer"),
    ):
        with pytest.raises(AnkiConnectError, match="Unable to reach AnkiConnect"):
            invoke("version")


def test_invoke_rejects_non_json_response_payload():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("no json")

    with patch("ankiops.anki_client._session.post", return_value=response):
        with pytest.raises(AnkiConnectError, match="non-JSON response"):
            invoke("version")

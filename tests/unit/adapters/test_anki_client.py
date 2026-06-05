"""Unit tests for the Anki HTTP connection client."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from ankiops.anki_client import AnkiConnectionError, invoke


def test_invoke_posts_to_ankiops_connect_and_returns_result():
    response = MagicMock()
    response.json.return_value = {"result": 1, "error": None}

    with patch("ankiops.anki_client._session.post", return_value=response) as post:
        assert invoke("version") == 1

    post.assert_called_once_with(
        "http://127.0.0.1:8766",
        json={"action": "version", "params": {}},
        timeout=10,
    )


def test_invoke_falls_back_to_anki_connect_for_standard_actions():
    anki_connect_response = MagicMock()
    anki_connect_response.json.return_value = {"result": 6, "error": None}

    with patch(
        "ankiops.anki_client._session.post",
        side_effect=[
            requests.ConnectionError("Connection reset by peer"),
            anki_connect_response,
        ],
    ) as post:
        assert invoke("version") == 6

    assert post.call_args_list[0].args == ("http://127.0.0.1:8766",)
    assert post.call_args_list[1].args == ("http://localhost:8765",)
    assert post.call_args_list[1].kwargs["json"] == {
        "action": "version",
        "version": 6,
        "params": {},
    }


def test_invoke_falls_back_when_ankiops_connect_lacks_standard_action():
    ankiops_connect_response = MagicMock()
    ankiops_connect_response.json.return_value = {
        "result": None,
        "error": "Unknown AnkiOpsConnect action: getActiveProfile",
    }
    anki_connect_response = MagicMock()
    anki_connect_response.json.return_value = {"result": "User 1", "error": None}

    with patch(
        "ankiops.anki_client._session.post",
        side_effect=[ankiops_connect_response, anki_connect_response],
    ) as post:
        assert invoke("getActiveProfile") == "User 1"

    assert post.call_args_list[0].args == ("http://127.0.0.1:8766",)
    assert post.call_args_list[1].args == ("http://localhost:8765",)


def test_invoke_requires_ankiops_connect_for_custom_actions():
    with patch(
        "ankiops.anki_client._session.post",
        side_effect=requests.ConnectionError("Connection reset by peer"),
    ) as post:
        with pytest.raises(AnkiConnectionError, match="requires AnkiOpsConnect"):
            invoke(
                "changeNotesNotetype",
                noteIds=[101],
                oldModel="AnkiOpsQA",
                newModel="collab/o/r/AnkiOpsQA",
            )

    post.assert_called_once()


def test_invoke_rejects_non_json_response_payload():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("no json")

    with patch("ankiops.anki_client._session.post", return_value=response):
        with pytest.raises(AnkiConnectionError, match="non-JSON response"):
            invoke("version")

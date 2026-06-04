from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from ankiops.ankiops_bridge import AnkiOpsBridgeClient, AnkiOpsBridgeError


def _response(payload):
    return SimpleNamespace(json=lambda: payload)


class _Session:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append((url, json, timeout))
        if self.error:
            raise self.error
        return self.response


def test_bridge_client_version_returns_result():
    session = _Session(_response({"result": {"version": 1}}))
    client = AnkiOpsBridgeClient(session=session)

    assert client.version() == {"version": 1}
    assert session.calls == [
        (
            "http://127.0.0.1:8766",
            {"action": "version", "params": {}},
            10,
        )
    ]


def test_bridge_client_change_notes_notetype_sends_feature_payload():
    session = _Session(_response({"result": {"changed": 2}}))
    client = AnkiOpsBridgeClient(session=session)

    assert client.change_notes_notetype(
        [101, 102],
        "AnkiOpsQA",
        "collab/o/r/AnkiOpsQA",
    ) == {"changed": 2}
    assert session.calls == [
        (
            "http://127.0.0.1:8766",
            {
                "action": "changeNotesNotetype",
                "params": {
                    "noteIds": [101, 102],
                    "oldModel": "AnkiOpsQA",
                    "newModel": "collab/o/r/AnkiOpsQA",
                },
            },
            10,
        )
    ]


def test_bridge_client_raises_structured_error():
    session = _Session(_response({"error": "boom", "result": None}))
    client = AnkiOpsBridgeClient(session=session)

    with pytest.raises(AnkiOpsBridgeError, match="boom"):
        client.version()


def test_bridge_client_raises_when_missing():
    session = _Session(error=requests.ConnectionError("refused"))
    client = AnkiOpsBridgeClient(session=session)

    with pytest.raises(AnkiOpsBridgeError, match="Unable to reach"):
        client.version()


def test_bridge_client_raises_on_malformed_response():
    response = SimpleNamespace(json=lambda: (_ for _ in ()).throw(ValueError()))
    session = _Session(response)
    client = AnkiOpsBridgeClient(session=session)

    with pytest.raises(AnkiOpsBridgeError, match="non-JSON"):
        client.version()

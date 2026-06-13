from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from anki_addon.host import AnkiOpsConnectHost


def test_host_dispatches_on_main_thread():
    collection = object()
    ran_on_main = []
    calls = []

    def run_on_main(work):
        ran_on_main.append(True)
        work()

    def dispatch_action(col, action, params):
        calls.append((col, action, params))
        return {"ok": True}

    host = AnkiOpsConnectHost(
        get_collection=lambda: collection,
        run_on_main=run_on_main,
        dispatch_action=dispatch_action,
    )

    assert host.dispatch_payload({"action": "version", "params": {"x": 1}}) == {
        "ok": True
    }
    assert ran_on_main == [True]
    assert calls == [(collection, "version", {"x": 1})]


def test_host_returns_structured_errors():
    host = AnkiOpsConnectHost(
        get_collection=lambda: None,
        run_on_main=lambda work: work(),
    )

    assert host.handle_payload({"action": "version", "params": {}}) == {
        "result": None,
        "error": "No Anki collection is open.",
    }


def test_host_validates_payload_shape():
    host = AnkiOpsConnectHost(
        get_collection=lambda: object(),
        run_on_main=lambda work: work(),
    )

    assert host.handle_payload({"action": "version", "params": []}) == {
        "result": None,
        "error": "AnkiOpsConnect requests require action and params.",
    }


def test_host_reads_json_request_body():
    host = AnkiOpsConnectHost(
        get_collection=lambda: object(),
        run_on_main=lambda work: work(),
    )
    body = b'{"action": "version", "params": {}}'
    handler = SimpleNamespace(
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        },
        rfile=io.BytesIO(body),
    )

    assert host.read_payload(handler) == {"action": "version", "params": {}}


def test_host_rejects_invalid_content_length():
    host = AnkiOpsConnectHost(
        get_collection=lambda: object(),
        run_on_main=lambda work: work(),
    )
    handler = SimpleNamespace(
        headers={
            "Content-Type": "application/json",
            "Content-Length": "chunked",
        },
        rfile=io.BytesIO(b""),
    )

    with pytest.raises(ValueError, match="Invalid Content-Length header"):
        host.read_payload(handler)

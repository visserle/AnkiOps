from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_bridge_host_module():
    addon_dir = Path(__file__).resolve().parents[2] / "anki_addon"
    package = sys.modules.get("anki_addon")
    if package is None:
        package = ModuleType("anki_addon")
        package.__path__ = [str(addon_dir)]
        sys.modules["anki_addon"] = package
    spec = importlib.util.spec_from_file_location(
        "anki_addon.bridge_host",
        addon_dir / "bridge_host.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bridge_host = _load_bridge_host_module()
AnkiOpsBridgeHost = bridge_host.AnkiOpsBridgeHost


def test_bridge_host_dispatches_on_main_thread():
    collection = object()
    ran_on_main = []
    calls = []

    def run_on_main(work):
        ran_on_main.append(True)
        work()

    def dispatch_action(col, action, params):
        calls.append((col, action, params))
        return {"ok": True}

    host = AnkiOpsBridgeHost(
        get_collection=lambda: collection,
        run_on_main=run_on_main,
        dispatch_action=dispatch_action,
    )

    assert host.dispatch_payload({"action": "version", "params": {"x": 1}}) == {
        "ok": True
    }
    assert ran_on_main == [True]
    assert calls == [(collection, "version", {"x": 1})]


def test_bridge_host_returns_structured_errors():
    host = AnkiOpsBridgeHost(
        get_collection=lambda: None,
        run_on_main=lambda work: work(),
    )

    assert host.handle_payload({"action": "version", "params": {}}) == {
        "result": None,
        "error": "No Anki collection is open.",
    }


def test_bridge_host_validates_payload_shape():
    host = AnkiOpsBridgeHost(
        get_collection=lambda: object(),
        run_on_main=lambda work: work(),
    )

    assert host.handle_payload({"action": "version", "params": []}) == {
        "result": None,
        "error": "Bridge requests require action and params.",
    }


def test_bridge_host_reads_json_request_body():
    host = AnkiOpsBridgeHost(
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


def test_bridge_host_rejects_invalid_content_length():
    host = AnkiOpsBridgeHost(
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

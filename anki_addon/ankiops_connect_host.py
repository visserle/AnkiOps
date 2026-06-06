"""HTTP host for AnkiOpsConnect."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .ankiops_connect_actions import dispatch_ankiops_connect_action

ANKIOPS_CONNECT_HOST = "127.0.0.1"
ANKIOPS_CONNECT_PORT = 8766
ANKIOPS_CONNECT_BODY_LIMIT = 1024 * 1024
ANKIOPS_CONNECT_MAIN_THREAD_TIMEOUT_SECONDS = 30


class _AnkiOpsConnectServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class AnkiOpsConnectHost:
    def __init__(
        self,
        *,
        get_collection: Callable[[], object | None],
        run_on_main: Callable[[Callable[[], None]], None],
        dispatch_action: Callable[
            [object, str, dict],
            object,
        ] = dispatch_ankiops_connect_action,
        host: str = ANKIOPS_CONNECT_HOST,
        port: int = ANKIOPS_CONNECT_PORT,
        body_limit: int = ANKIOPS_CONNECT_BODY_LIMIT,
        timeout_seconds: int = ANKIOPS_CONNECT_MAIN_THREAD_TIMEOUT_SECONDS,
    ) -> None:
        self._get_collection = get_collection
        self._run_on_main = run_on_main
        self._dispatch_action = dispatch_action
        self._host = host
        self._port = port
        self._body_limit = body_limit
        self._timeout_seconds = timeout_seconds
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> bool:
        if self._server is not None:
            return True
        try:
            server = _AnkiOpsConnectServer(
                (self._host, self._port),
                self._handler_class(),
            )
        except OSError as error:
            print(f"AnkiOpsConnect could not start: {error}")
            return False
        thread = threading.Thread(
            target=server.serve_forever,
            name="AnkiOpsConnect",
            daemon=True,
        )
        thread.start()
        self._server = server
        return True

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.dispatch_payload(payload)
            return {"result": result, "error": None}
        except Exception as error:
            return {"result": None, "error": str(error)}

    def dispatch_payload(self, payload: dict[str, Any]):
        action = payload.get("action")
        params = payload.get("params", {})
        if not isinstance(action, str) or not isinstance(params, dict):
            raise ValueError("AnkiOpsConnect requests require action and params.")
        return self._run_action_on_main(action, params)

    def read_payload(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        content_type = handler.headers.get("Content-Type", "").split(";")[0]
        if content_type != "application/json":
            raise ValueError("AnkiOpsConnect requests must use application/json.")
        raw_length = handler.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Missing Content-Length.")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if length > self._body_limit:
            raise ValueError("AnkiOpsConnect request body is too large.")
        raw = handler.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AnkiOpsConnect request body must be a JSON object.")
        return payload

    def _run_action_on_main(self, action: str, params: dict[str, Any]):
        done = threading.Event()
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                collection = self._get_collection()
                if collection is None:
                    raise RuntimeError("No Anki collection is open.")
                box["result"] = self._dispatch_action(collection, action, params)
            except Exception as error:
                box["error"] = str(error)
            finally:
                done.set()

        self._run_on_main(work)
        if not done.wait(timeout=self._timeout_seconds):
            raise RuntimeError("Timed out waiting for Anki main thread.")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("result")

    def _handler_class(self):
        ankiops_connect_host = self

        class _AnkiOpsConnectHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    payload = ankiops_connect_host.read_payload(self)
                    response = ankiops_connect_host.handle_payload(payload)
                except Exception as error:
                    response = {"result": None, "error": str(error)}
                self._send_json(response)

            def log_message(self, _format: str, *_args) -> None:
                return

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return _AnkiOpsConnectHandler

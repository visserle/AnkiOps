"""Client for the AnkiOps add-on bridge."""

from __future__ import annotations

from typing import Any

import requests

ANKIOPS_BRIDGE_URL = "http://127.0.0.1:8766"
CHANGE_NOTETYPE_ACTION = "changeNotesNotetype"

DEFAULT_TIMEOUT_SECONDS = 10


class AnkiOpsBridgeError(Exception):
    """Raised when the AnkiOps add-on bridge is unavailable or returns an error."""


class AnkiOpsBridgeClient:
    def __init__(
        self,
        *,
        url: str = ANKIOPS_BRIDGE_URL,
        session: requests.Session | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = url
        self._session = session or requests.Session()
        self._timeout = timeout

    def version(self) -> dict[str, Any]:
        return self._invoke("version", {})

    def change_notes_notetype(
        self,
        note_ids: list[int],
        old_model: str,
        new_model: str,
    ) -> dict[str, Any]:
        return self._invoke(
            CHANGE_NOTETYPE_ACTION,
            {
                "noteIds": note_ids,
                "oldModel": old_model,
                "newModel": new_model,
            },
        )

    def _invoke(self, action: str, params: dict[str, Any]) -> Any:
        try:
            response = self._session.post(
                self._url,
                json={"action": action, "params": params},
                timeout=self._timeout,
            )
        except requests.RequestException as error:
            raise AnkiOpsBridgeError(
                f"Unable to reach AnkiOps bridge at {self._url}: {error}"
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise AnkiOpsBridgeError(
                "AnkiOps bridge returned a non-JSON response."
            ) from error

        if payload.get("error"):
            raise AnkiOpsBridgeError(str(payload["error"]))
        if "result" not in payload:
            raise AnkiOpsBridgeError("AnkiOps bridge response missing 'result' field.")
        return payload["result"]


_default_client = AnkiOpsBridgeClient()


def change_notes_notetype(
    note_ids: list[int],
    old_model: str,
    new_model: str,
) -> dict[str, Any]:
    return _default_client.change_notes_notetype(note_ids, old_model, new_model)

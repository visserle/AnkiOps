"""AnkiConnect HTTP client."""

import logging
from typing import Any

import requests

ANKI_CONNECT_URL = "http://localhost:8765"

logger = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"Connection": "close"})


class AnkiConnectError(Exception):
    """Raised when AnkiConnect returns an error response."""


def invoke(action: str, **params) -> Any:
    """Send a request to AnkiConnect and return the result.

    Raises AnkiConnectError when AnkiConnect returns an error.
    """
    try:
        response = _session.post(
            ANKI_CONNECT_URL,
            json={"action": action, "version": 6, "params": params},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise AnkiConnectError(
            f"Unable to reach AnkiConnect at {ANKI_CONNECT_URL}: {error}"
        ) from error

    try:
        result = response.json()
    except ValueError as error:
        raise AnkiConnectError("AnkiConnect returned a non-JSON response.") from error

    if not isinstance(result, dict):
        raise AnkiConnectError("AnkiConnect returned a malformed response payload.")
    if result.get("error"):
        raise AnkiConnectError(result["error"])
    if "result" not in result:
        raise AnkiConnectError("AnkiConnect response missing 'result' field.")
    return result["result"]

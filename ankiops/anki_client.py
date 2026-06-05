"""HTTP client for the local AnkiOps add-on bridge."""

from typing import Any

import requests

ANKIOPS_BRIDGE_URL = "http://127.0.0.1:8766"

_session = requests.Session()


class AnkiConnectError(Exception):
    """Raised when Anki returns an error response."""


def invoke(action: str, **params) -> Any:
    """Send a request to the AnkiOps add-on bridge and return the result.

    Raises AnkiConnectError when Anki returns an error.
    """
    try:
        response = _session.post(
            ANKIOPS_BRIDGE_URL,
            json={"action": action, "params": params},
            timeout=10,
        )
    except requests.RequestException as error:
        raise AnkiConnectError(
            f"Unable to reach AnkiOps bridge at {ANKIOPS_BRIDGE_URL}: {error}"
        ) from error

    try:
        result = response.json()
    except ValueError as error:
        raise AnkiConnectError(
            "AnkiOps bridge returned a non-JSON response."
        ) from error

    if result.get("error"):
        raise AnkiConnectError(result["error"])
    if "result" not in result:
        raise AnkiConnectError("AnkiOps bridge response missing 'result' field.")
    return result["result"]

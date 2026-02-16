"""AnkiConnect HTTP client."""

import logging
from typing import Any

import requests

ANKI_CONNECT_URL = "http://localhost:8765"

logger = logging.getLogger(__name__)

_session = requests.Session()


class AnkiConnectError(Exception):
    """Raised when AnkiConnect returns an error response."""


def invoke(action: str, **params) -> Any:
    """Send a request to AnkiConnect and return the result.

    Raises AnkiConnectError when AnkiConnect returns an error.
    """
    response = _session.post(
        ANKI_CONNECT_URL,
        json={"action": action, "version": 6, "params": params},
        timeout=10,
    )
    result = response.json()
    if result.get("error"):
        raise AnkiConnectError(result["error"])
    return result["result"]

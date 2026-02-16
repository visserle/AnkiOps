"""AnkiConnect HTTP client."""

import logging
from typing import Any

import requests

ANKI_CONNECT_URL = "http://localhost:8765"

logger = logging.getLogger(__name__)


def invoke(action: str, **params) -> Any:
    """Send a request to AnkiConnect and return the result.

    Raises an Exception when AnkiConnect returns an error.
    """
    response = requests.post(
        ANKI_CONNECT_URL,
        json={"action": action, "version": 6, "params": params},
        timeout=10,
    )
    result = response.json()
    if result.get("error"):
        raise Exception(f"AnkiConnect error: {result['error']}")
    return result["result"]

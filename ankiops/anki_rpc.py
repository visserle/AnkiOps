"""HTTP client for AnkiOpsConnect with AnkiConnect fallback."""

import os
from typing import Any

import requests

DEFAULT_ANKIOPS_CONNECT_URL = "http://127.0.0.1:8766"
ANKIOPS_CONNECT_URL_ENV = "ANKIOPS_CONNECT_URL"
ANKI_CONNECT_URL = "http://localhost:8765"
DEFAULT_TIMEOUT_SECONDS = 10
ANKIOPS_CONNECT_ONLY_ACTIONS = frozenset({"convertNotesToNoteType"})

_session = requests.Session()


class AnkiConnectionError(Exception):
    """Raised when Anki returns an error response."""


class _AnkiOpsConnectUnavailable(AnkiConnectionError):
    pass


class _AnkiOpsConnectUnsupportedAction(AnkiConnectionError):
    pass


def _ankiops_connect_url() -> tuple[str, bool]:
    url = os.environ.get(ANKIOPS_CONNECT_URL_ENV)
    if url is None:
        return DEFAULT_ANKIOPS_CONNECT_URL, False
    return url, True


def invoke(action: str, **params) -> Any:
    """Send a request through AnkiOpsConnect, falling back to AnkiConnect.

    Raises AnkiConnectionError when Anki returns an error.
    """
    ankiops_url, is_custom = _ankiops_connect_url()
    try:
        return _invoke_ankiops_connect(action, params, ankiops_url)
    except (
        _AnkiOpsConnectUnavailable,
        _AnkiOpsConnectUnsupportedAction,
    ) as ankiops_connect_error:
        if is_custom:
            raise
        if action in ANKIOPS_CONNECT_ONLY_ACTIONS:
            raise AnkiConnectionError(
                f"{action} requires AnkiOpsConnect. AnkiConnect is available "
                "only for standard Anki actions."
            ) from ankiops_connect_error
        try:
            return _invoke_anki_connect(action, params)
        except AnkiConnectionError as anki_connect_error:
            raise AnkiConnectionError(
                "Unable to complete request with AnkiOpsConnect or AnkiConnect. "
                f"AnkiOpsConnect: {ankiops_connect_error}; "
                f"AnkiConnect: {anki_connect_error}"
            ) from anki_connect_error


def _invoke_ankiops_connect(
    action: str,
    params: dict[str, Any],
    ankiops_connect_url: str,
) -> Any:
    try:
        response = _session.post(
            ankiops_connect_url,
            json={"action": action, "params": params},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise _AnkiOpsConnectUnavailable(
            f"Unable to reach AnkiOpsConnect at {ankiops_connect_url}: {error}"
        ) from error

    try:
        result = response.json()
    except ValueError as error:
        raise _AnkiOpsConnectUnavailable(
            "AnkiOpsConnect returned a non-JSON response."
        ) from error

    if result.get("error"):
        error = str(result["error"])
        if error.startswith("Unknown AnkiOpsConnect action:"):
            raise _AnkiOpsConnectUnsupportedAction(error)
        raise AnkiConnectionError(error)
    if "result" not in result:
        raise _AnkiOpsConnectUnavailable(
            "AnkiOpsConnect response missing 'result' field."
        )
    return result["result"]


def _invoke_anki_connect(action: str, params: dict[str, Any]) -> Any:
    try:
        response = _session.post(
            ANKI_CONNECT_URL,
            json={"action": action, "version": 6, "params": params},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise AnkiConnectionError(
            f"Unable to reach AnkiConnect at {ANKI_CONNECT_URL}: {error}"
        ) from error

    try:
        result = response.json()
    except ValueError as error:
        raise AnkiConnectionError(
            "AnkiConnect returned a non-JSON response."
        ) from error

    if result.get("error"):
        raise AnkiConnectionError(result["error"])
    if "result" not in result:
        raise AnkiConnectionError("AnkiConnect response missing 'result' field.")
    return result["result"]

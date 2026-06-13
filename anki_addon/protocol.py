"""Shared AnkiOpsConnect protocol vocabulary."""

from __future__ import annotations

from collections.abc import Callable

ANKIOPS_CONNECT_VERSION = 1
ANKIOPS_KEY_FIELD_NAME = "AnkiOps Key"

AnkiOpsConnectAction = Callable[[object, dict], object]


class AnkiOpsConnectActionError(Exception):
    """Raised when an AnkiOpsConnect action cannot be completed safely."""

"""Shared AnkiOpsConnect protocol vocabulary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ANKIOPS_CONNECT_VERSION = 1
ANKIOPS_KEY_FIELD_NAME = "AnkiOps Key"

AnkiOpsConnectAction = Callable[[Any, dict[str, Any]], Any]


class AnkiOpsConnectActionError(Exception):
    """Raised when an AnkiOpsConnect action cannot be completed safely."""

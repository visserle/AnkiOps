"""Shared CLI helpers for Anki connectivity."""

import logging

from ankiops.anki import AnkiAdapter

logger = logging.getLogger(__name__)


def connect_or_exit() -> AnkiAdapter:
    """Verify an Anki HTTP connection is reachable; exit on failure."""
    anki = AnkiAdapter()
    try:
        version = anki.get_version()
        logger.debug(f"Connected to Anki HTTP API (version {version})")
    except Exception as error:
        logger.error(
            "Error connecting to Anki. Make sure Anki is running and either "
            "AnkiOpsConnect or AnkiConnect is enabled."
        )
        logger.debug(f"Connection error details: {error}")
        raise SystemExit(1)
    return anki

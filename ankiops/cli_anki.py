"""Shared CLI helpers for Anki connectivity."""

import logging

from ankiops.anki import AnkiAdapter

logger = logging.getLogger(__name__)


def connect_or_exit() -> AnkiAdapter:
    """Verify the AnkiOps add-on bridge is reachable; exit on failure."""
    anki = AnkiAdapter()
    try:
        version = anki.get_version()
        logger.debug(f"Connected to AnkiOps add-on bridge (version {version})")
    except Exception as error:
        logger.error(
            "Error connecting to the AnkiOps add-on. Make sure Anki is running and "
            "the AnkiOps add-on is enabled."
        )
        logger.debug(f"Connection error details: {error}")
        raise SystemExit(1)
    return anki

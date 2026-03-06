"""Shared CLI helpers for Anki connectivity."""

import logging

from ankiops.anki import AnkiAdapter

logger = logging.getLogger(__name__)


def connect_or_exit() -> AnkiAdapter:
    """Verify AnkiConnect is reachable; exit on failure."""
    anki = AnkiAdapter()
    try:
        version = anki.get_version()
        logger.debug(f"Connected to AnkiConnect (version {version})")
    except Exception as error:
        logger.error(f"Error connecting to AnkiConnect: {error}")
        logger.error("Make sure Anki is running and AnkiConnect is installed.")
        raise SystemExit(1)
    return anki

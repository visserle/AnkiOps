"""Use Case: Synchronize Note Types to Anki."""

import logging
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.fs import FileSystemAdapter

logger = logging.getLogger(__name__)


def sync_note_types(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    note_types_dir: Path,
) -> str | None:
    """Ensure all Note Types defined in 'note_types_dir' exist in Anki.

    Returns a human-readable summary string, or None if nothing happened.
    """
    configs = fs_port.load_note_type_configs(note_types_dir)
    if not configs:
        logger.debug(f"No note types found in {note_types_dir}")
        return None

    existing = set(anki_port.fetch_model_names())

    to_create = [c for c in configs if c.name not in existing]
    to_update = [c for c in configs if c.name in existing]

    parts = []

    if to_create:
        names = ", ".join(c.name for c in to_create)
        logger.info(f"Note types: {len(to_create)} created ({names})")
        anki_port.create_models(to_create)
        parts.append(f"{len(to_create)} created")

    if to_update:
        logger.debug(f"Checking {len(to_update)} existing note types for updates...")
        states = anki_port.fetch_model_states([c.name for c in to_update])
        anki_port.update_models(to_update, states)
        names = ", ".join(c.name for c in to_update)
        logger.debug(f"Note types: {len(to_update)} synced ({names})")
        parts.append(f"{len(to_update)} synced")

    if not parts:
        logger.debug("Note types: up to date")
        return None

    return ", ".join(parts)

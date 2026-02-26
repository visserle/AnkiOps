"""Use Case: Synchronize Note Types to Anki."""

import hashlib
import json
import logging
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter

logger = logging.getLogger(__name__)


def _note_types_sync_hash(configs) -> str:
    payload = []
    for config in sorted(configs, key=lambda config_item: config_item.name):
        payload.append(
            {
                "name": config.name,
                "is_cloze": config.is_cloze,
                "is_choice": config.is_choice,
                "css": config.css,
                "fields": [
                    {
                        "name": field.name,
                        "prefix": field.prefix,
                        "identifying": field.identifying,
                    }
                    for field in config.fields
                ],
                "templates": [
                    {
                        "Name": template.get("Name", ""),
                        "Front": template.get("Front", ""),
                        "Back": template.get("Back", ""),
                    }
                    for template in config.templates
                ],
            }
        )

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _note_types_names_signature(configs) -> str:
    return ",".join(sorted(config.name for config in configs))


def sync_note_types(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    note_types_dir: Path,
    db_port: SQLiteDbAdapter | None = None,
) -> str | None:
    """Ensure all Note Types defined in 'note_types_dir' exist in Anki.

    Returns a human-readable summary string, or None if nothing happened.
    """
    configs = fs_port.load_note_type_configs(note_types_dir)
    if not configs:
        logger.debug(f"No note types found in {note_types_dir}")
        return None

    existing = set(anki_port.fetch_model_names())
    to_create = [config for config in configs if config.name not in existing]
    to_update = [config for config in configs if config.name in existing]

    local_hash = _note_types_sync_hash(configs)
    names_signature = _note_types_names_signature(configs)
    cached_state = db_port.get_note_type_sync_state() if db_port is not None else None
    if (
        db_port is not None
        and not to_create
        and cached_state == (local_hash, names_signature)
    ):
        logger.debug(
            "Note types unchanged since last successful sync; skipping model diff"
        )
        return f"{len(to_update)} up to date (cached)"

    parts = []

    if to_create:
        names = ", ".join(config.name for config in to_create)
        logger.info(f"Note types: {len(to_create)} created ({names})")
        anki_port.create_models(to_create)
        parts.append(f"{len(to_create)} created")

    if to_update:
        logger.debug(f"Checking {len(to_update)} existing note types for updates...")
        states = anki_port.fetch_model_states([config.name for config in to_update])
        anki_port.update_models(to_update, states)
        names = ", ".join(config.name for config in to_update)
        logger.debug(f"Note types: {len(to_update)} synced ({names})")
        parts.append(f"{len(to_update)} synced")

    if not parts:
        logger.debug("Note types: up to date")
        return None

    if db_port is not None:
        db_port.set_note_type_sync_state(local_hash, names_signature)

    return ", ".join(parts)

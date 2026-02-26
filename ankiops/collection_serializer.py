"""Serialize and deserialize AnkiOps collections to/from JSON format."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ankiops.config import (
    ANKIOPS_DB,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
    get_collection_dir,
    get_note_types_dir,
)
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)


def serialize_collection(collection_dir: Path) -> dict[str, Any]:
    """Serialize the collection into an in-memory JSON-compatible mapping."""
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an AnkiOps collection: {collection_dir}")

    fs = FileSystemAdapter()
    fs.load_note_type_configs(get_note_types_dir())

    serialized_data: dict[str, Any] = {
        "collection": {
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    errors = []
    md_files = fs.find_markdown_files(collection_dir)

    for md_file in md_files:
        try:
            parsed = fs.read_markdown_file(md_file)
        except Exception as error:
            errors.append(f"Error parsing {md_file.name}: {error}")
            continue

        deck_data: dict[str, Any] = {
            "name": file_stem_to_deck_name(md_file.stem),
            "notes": [],
        }
        notes_data: list[dict[str, Any]] = deck_data["notes"]
        for note in parsed.notes:
            notes_data.append(
                {
                    "note_key": note.note_key,
                    "note_type": note.note_type,
                    "fields": note.fields,
                }
            )

        if notes_data:
            decks_data = serialized_data["decks"]
            if isinstance(decks_data, list):
                decks_data.append(deck_data)

    total_notes = sum(
        len(deck.get("notes", []))
        for deck in serialized_data["decks"]
        if isinstance(deck, dict)
    )
    total_decks = len(serialized_data["decks"])
    logger.debug(f"Serialized {total_decks} deck(s), {total_notes} note(s) in memory")

    if errors:
        logger.warning(
            f"Serialization completed with {len(errors)} error(s). "
            "Some notes were skipped. Review errors above."
        )

    return serialized_data


def serialize_collection_to_json(
    collection_dir: Path,
    output_file: Path,
) -> dict[str, Any]:
    """Serialize entire collection to JSON format.

    Args:
        collection_dir: Path to the collection directory
        output_file: Path where JSON file will be written

    Returns:
        Dictionary containing the serialized data
    """
    serialized_data = serialize_collection(collection_dir)
    total_notes = sum(
        len(deck.get("notes", []))
        for deck in serialized_data["decks"]
        if isinstance(deck, dict)
    )
    total_decks = len(serialized_data["decks"])

    with output_file.open("w", encoding="utf-8") as output_handle:
        json.dump(serialized_data, output_handle, indent=2, ensure_ascii=False)

    logger.info(
        f"Serialized {total_decks} deck(s), {total_notes} note(s) to {output_file}"
    )

    return serialized_data


def deserialize_collection_from_json(
    json_file: Path,
    overwrite: bool = False,
) -> None:
    """Deserialize collection from JSON format.

    In development mode (pyproject.toml with name="ankiops" in cwd),
    unpacks to ./collection. Otherwise, unpacks to the current working directory.

    Args:
        json_file: Path to JSON file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
    """
    with json_file.open("r", encoding="utf-8") as input_handle:
        data = json.load(input_handle)

    logger.debug(f"Importing serialized collection from: {json_file}")
    deserialize_collection_data(data, overwrite=overwrite)


def deserialize_collection_data(
    data: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Deserialize collection from an in-memory JSON-compatible mapping."""
    root_dir = get_collection_dir()
    logger.debug(f"Target directory: {root_dir}")

    if not isinstance(data, dict):
        raise ValueError("Serialized data must be a JSON object mapping")

    decks = data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Serialized data must contain a top-level 'decks' list")

    if not overwrite:
        existing_md_files = list(root_dir.glob("*.md"))
        if existing_md_files:
            logger.warning(
                f"Found {len(existing_md_files)} existing markdown file(s) "
                f"in {root_dir}. Use --overwrite to replace them."
            )

    fs = FileSystemAdapter()
    configs = fs.load_note_type_configs(get_note_types_dir())
    config_by_name = {config.name: config for config in configs}

    total_decks = 0
    total_notes = 0
    skipped_notes = 0

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            continue

        filename = deck_name_to_file_stem(deck_name) + ".md"
        output_path = root_dir / filename

        lines = []
        written_notes = 0

        for note in notes:
            note_key = note.get("note_key")
            fields = note["fields"]

            note_type = note.get("note_type")
            config = (
                config_by_name.get(note_type) if isinstance(note_type, str) else None
            )

            if config is None:
                try:
                    note_type = fs._infer_note_type(fields)
                except ValueError as error:
                    logger.warning(
                        f"Cannot infer note type in deck '{deck_name}': {error}, "
                        "skipping note"
                    )
                    skipped_notes += 1
                    continue
                config = config_by_name.get(note_type)

            if config is None:
                logger.warning(
                    f"Unknown note type '{note_type}' in deck '{deck_name}', "
                    "skipping note"
                )
                skipped_notes += 1
                continue

            if note_key:
                lines.append(f"<!-- note_key: {note_key} -->")

            for field in config.fields:
                field_content = fields.get(field.name)
                if field_content and field.prefix:
                    lines.append(f"{field.prefix} {field_content}")
            written_notes += 1

            lines.append("")
            lines.append("---")
            lines.append("")

        # Remove trailing separator
        while lines and lines[-1] in ("", "---"):
            lines.pop()

        content = "\n".join(lines)
        if overwrite or not output_path.exists():
            output_path.write_text(content, encoding="utf-8")
            logger.info(
                f"  Created {clickable_path(output_path)} ({written_notes} notes)"
            )
        else:
            logger.warning(
                f"Skipped {clickable_path(output_path)} "
                "(already exists, use --overwrite to replace)"
            )

        total_decks += 1
        total_notes += written_notes

    logger.info(
        f"Deserialized {total_decks} deck(s), {total_notes} note(s) to {root_dir}"
    )
    if skipped_notes:
        logger.warning(
            f"Skipped {skipped_notes} note(s) due to missing/invalid note type metadata"
        )

    db_path = root_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.info(
            "Run 'ankiops init' to set up this collection with your Anki profile."
        )

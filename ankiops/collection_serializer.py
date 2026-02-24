"""Serialize and deserialize AnkiOps collections to/from JSON format."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ankiops.config import ANKIOPS_DB, get_collection_dir, get_note_types_dir
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)


def serialize_collection_to_json(
    collection_dir: Path,
    output_file: Path,
) -> dict:
    """Serialize entire collection to JSON format.

    Args:
        collection_dir: Path to the collection directory
        output_file: Path where JSON file will be written

    Returns:
        Dictionary containing the serialized data
    """
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an AnkiOps collection: {collection_dir}")

    fs = FileSystemAdapter()
    fs.load_note_type_configs(get_note_types_dir())

    serialized_data = {
        "collection": {
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    errors = []
    md_files = fs.find_markdown_files(collection_dir)
    logger.debug(f"Found {len(md_files)} deck file(s) to serialize")

    for md_file in md_files:
        logger.debug(f"Processing {md_file.name}...")

        try:
            parsed = fs.read_markdown_file(md_file)
        except Exception as e:
            errors.append(f"Error parsing {md_file.name}: {e}")
            continue

        deck_data = {}

        deck_data["name"] = md_file.stem.replace("__", "::")
        deck_data["notes"] = []

        for note in parsed.notes:
            note_data = {}
            note_data["note_key"] = note.note_key
            note_data["note_type"] = note.note_type
            note_data["fields"] = note.fields
            deck_data["notes"].append(note_data)

        if deck_data["notes"]:
            serialized_data["decks"].append(deck_data)

    total_notes = sum(len(deck["notes"]) for deck in serialized_data["decks"])
    total_decks = len(serialized_data["decks"])

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Serialized {total_decks} deck(s), {total_notes} note(s) to {output_file}"
    )

    if errors:
        logger.warning(
            f"Serialization completed with {len(errors)} error(s). "
            "Some notes were skipped. Review errors above."
        )

    return serialized_data


def deserialize_collection_from_json(
    json_file: Path, overwrite: bool = False,
) -> None:
    """Deserialize collection from JSON format.

    In development mode (pyproject.toml with name="ankiops" in cwd),
    unpacks to ./collection. Otherwise, unpacks to the current working directory.

    Args:
        json_file: Path to JSON file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
    """
    root_dir = get_collection_dir()

    logger.debug(f"Importing serialized collection from: {json_file}")
    logger.debug(f"Target directory: {root_dir}")

    if not overwrite:
        existing_md_files = list(root_dir.glob("*.md"))
        if existing_md_files:
            logger.warning(
                f"Found {len(existing_md_files)} existing markdown file(s) "
                f"in {root_dir}. Use --overwrite to replace them."
            )

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fs = FileSystemAdapter()
    configs = fs.load_note_type_configs(get_note_types_dir())
    config_by_name = {c.name: c for c in configs}

    total_decks = 0
    total_notes = 0

    for deck in data["decks"]:
        deck_name = deck["name"]
        notes = deck["notes"]

        filename = deck_name.replace("::", "__") + ".md"
        output_path = root_dir / filename

        lines = []

        for note in notes:
            note_key = note.get("note_key")
            fields = note["fields"]

            note_type = note.get("note_type")
            if not note_type:
                try:
                    note_type = fs._infer_note_type(fields)
                except ValueError as e:
                    logger.warning(
                        f"Cannot infer note type in deck '{deck_name}': {e}, skipping"
                    )
                    continue

            if note_key:
                lines.append(f"<!-- note_key: {note_key} -->")

            config = config_by_name.get(note_type)
            if config:
                for field in config.fields:
                    field_content = fields.get(field.name)
                    if field_content and field.prefix:
                        lines.append(f"{field.prefix} {field_content}")

            lines.append("")
            lines.append("---")
            lines.append("")

        # Remove trailing separator
        while lines and lines[-1] in ("", "---"):
            lines.pop()

        content = "\n".join(lines)
        if overwrite or not output_path.exists():
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"  Created {clickable_path(output_path)} ({len(notes)} notes)")
        else:
            logger.debug(
                f"Skipped {clickable_path(output_path)} "
                "(already exists, use --overwrite to replace)"
            )

        total_decks += 1
        total_notes += len(notes)

    logger.info(
        f"Deserialized {total_decks} deck(s), {total_notes} note(s) to {root_dir}"
    )

    db_path = root_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.info(
            "Run 'ankiops init' to set up this collection with your Anki profile."
        )

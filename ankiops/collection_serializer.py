"""Serialize and deserialize AnkiOps collections to/from JSON format."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR, MARKER_FILE, get_collection_dir
from ankiops.log import clickable_path
from ankiops.models import FileState, Note
from ankiops.note_type_config import registry

logger = logging.getLogger(__name__)

# Regex patterns for media file references
MARKDOWN_IMAGE_PATTERN = r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?"
ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
HTML_IMG_PATTERN = r'<img[^>]+src=["\']([^"\']+)["\']'




def _normalize_media_path(path: str) -> str:
    """Normalize media path by stripping angle brackets and media/ prefix.

    Args:
        path: Raw path string from markdown/HTML

    Returns:
        Normalized path without angle brackets or media/ prefix
    """
    path = path.strip("<>")
    if path.startswith(f"{LOCAL_MEDIA_DIR}/"):
        path = path[len(LOCAL_MEDIA_DIR) + 1 :]
    return path


def extract_media_references(text: str) -> set[str]:
    """Extract media file references from markdown text.

    Finds:
    - Markdown images: ![alt](filename.png)
    - Anki sound: [sound:audio.mp3]
    - HTML img tags: <img src="file.jpg">

    Returns:
        Set of normalized media file paths (without media/ prefix)
    """
    media_files = set()

    # Extract from all three pattern types
    for pattern in [MARKDOWN_IMAGE_PATTERN, ANKI_SOUND_PATTERN, HTML_IMG_PATTERN]:
        for match in re.finditer(pattern, text):
            path = _normalize_media_path(match.group(1))
            if path:  # Only add if path is not empty
                media_files.add(path)

    return media_files


def serialize_collection_to_json(
    collection_dir: Path,
    output_file: Path,
    no_ids: bool = False,
) -> dict:
    """Serialize entire collection to JSON format.

    Args:
        collection_dir: Path to the collection directory
        output_file: Path where JSON file will be written
        no_ids: If True, exclude note_id and deck_id from serialized output

    Returns:
        Dictionary containing the serialized data
    """
    # Read collection config
    marker_path = collection_dir / MARKER_FILE
    if not marker_path.exists():
        raise ValueError(f"Not a AnkiOps collection: {collection_dir}")

    # Use local project media directory
    media_dir_path = collection_dir / LOCAL_MEDIA_DIR

    # Build JSON structure
    serialized_data = {
        "collection": {
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    # Track all media files referenced in notes
    all_media_files = set()
    # Track errors during serialization
    errors = []

    # Process all markdown files in collection
    md_files = sorted(collection_dir.glob("*.md"))
    logger.debug(f"Found {len(md_files)} deck file(s) to serialize")

    for md_file in md_files:
        logger.debug(f"Processing {md_file.name}...")
        content = md_file.read_text()

        # Extract deck_id and remaining content
        deck_id, remaining_content = FileState.extract_deck_id(content)

        # Split into note blocks
        note_blocks_raw = remaining_content.split("\n\n---\n\n")

        # Build deck data with deck_id first (matching markdown convention)
        deck_data = {}
        if not no_ids:
            deck_data["deck_id"] = str(deck_id) if deck_id else None
        deck_data["name"] = md_file.stem.replace("__", "::")  # Restore :: from __
        deck_data["notes"] = []

        line_number = 1  # Track for error reporting
        for block_text in note_blocks_raw:
            block_text = block_text.strip()
            if not block_text:
                continue

            try:
                parsed = Note.from_block(block_text)

                # Convert to JSON-friendly format with note_id first (matching markdown)
                note_data = {}
                if not no_ids:
                    note_data["note_id"] = (
                        str(parsed.note_id) if parsed.note_id else None
                    )
                note_data["fields"] = parsed.fields

                deck_data["notes"].append(note_data)
                line_number += block_text.count("\n") + 3  # +3 for the separator

            except Exception as e:
                error_msg = (
                    f"Error parsing note in {md_file.name} at line {line_number}: {e}"
                )
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue processing other notes
                line_number += block_text.count("\n") + 3

        if deck_data["notes"]:
            serialized_data["decks"].append(deck_data)

    total_notes = sum(len(deck["notes"]) for deck in serialized_data["decks"])
    total_decks = len(serialized_data["decks"])

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Serialized {total_decks} deck(s), {total_notes} note(s) to {output_file}"
    )

    # Report any errors encountered during serialization
    if errors:
        logger.warning(
            f"Serialization completed with {len(errors)} error(s). "
            "Some notes were skipped. Review errors above."
        )

    return serialized_data


def deserialize_collection_from_json(
    json_file: Path, overwrite: bool = False, no_ids: bool = False
) -> None:
    """Deserialize collection from JSON format.

    In development mode (pyproject.toml with name="ankiops" in cwd),
    unpacks to ./collection. Otherwise, unpacks to the current working directory.

    Note: This only extracts markdown files. Run 'ankiops init' after
    deserializing to set up the .ankiops config file with your profile settings.

    Args:
        json_file: Path to JSON file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
        no_ids: If True, skip writing deck_id and note_id comments to markdown
    """
    # Use collection directory (respects development mode)
    root_dir = get_collection_dir()

    logger.debug(f"Importing serialized collection from: {json_file}")
    logger.debug(f"Target directory: {root_dir}")

    # Check for existing markdown files that would be overwritten
    if not overwrite:
        existing_md_files = list(root_dir.glob("*.md"))
        if existing_md_files:
            logger.warning(
                f"Found {len(existing_md_files)} existing markdown file(s) "
                f"in {root_dir}. Use --overwrite to replace them."
            )

    # Load JSON data directly
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each deck
    total_decks = 0
    total_notes = 0

    for deck in data["decks"]:
        deck_name = deck["name"]
        deck_id = deck.get("deck_id")
        notes = deck["notes"]

        # Sanitize filename (replace :: with __)
        filename = deck_name.replace("::", "__") + ".md"
        output_path = root_dir / filename

        # Build markdown content
        lines = []

        # Add deck_id if present and not ignoring IDs
        if deck_id and not no_ids:
            lines.append(f"<!-- deck_id: {deck_id} -->")

        # Process each note
        for note in notes:
            note_id = note.get("note_id")
            fields = note["fields"]

            # Infer note type from fields
            try:
                note_type = Note.infer_note_type(fields)
            except ValueError as e:
                logger.warning(
                    f"Cannot infer note type in deck '{deck_name}': {e}, skipping note"
                )
                continue

            # Add note_id if present and not ignoring IDs
            if note_id and not no_ids:
                lines.append(f"<!-- note_id: {note_id} -->")

            # Get field mappings for this note type
            # Use registry.note_config to get legacy list of (field_name, prefix)
            note_config = registry.note_config.get(note_type)
            if not note_config:
                logger.warning(
                    f"Unknown note type '{note_type}' "
                    f"in deck '{deck_name}', skipping note"
                )
                continue

            # Format fields according to note type configuration
            for field_name, prefix in note_config:
                field_content = fields.get(field_name)
                if field_content:
                    lines.append(f"{prefix} {field_content}")

            # Add separator between notes
            lines.append("")
            lines.append("---")
            lines.append("")

        # Remove trailing separator
        while lines and lines[-1] in ("", "---"):
            lines.pop()

        # Write file
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
        f"Deserialized {total_decks} deck(s), {total_notes} note(s)"
        f" to {root_dir}"
    )

    # Check if .ankiops marker file exists
    marker_path = root_dir / MARKER_FILE
    if not marker_path.exists():
        logger.info(
            "Run 'ankiops init' to set up this collection with your Anki profile."
        )

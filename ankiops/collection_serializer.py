"""Serialize and deserialize AnkiOps collections to/from JSON format."""

import hashlib
import json
import logging
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ankiops.config import MARKER_FILE, NOTE_CONFIG, get_collection_dir
from ankiops.log import clickable_path
from ankiops.models import FileState, Note

logger = logging.getLogger(__name__)

# Regex patterns for media file references
MARKDOWN_IMAGE_PATTERN = r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?"
ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
HTML_IMG_PATTERN = r'<img[^>]+src=["\']([^"\']+)["\']'


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file. Returns hexadecimal hash string."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_zipfile_hash(zipf: zipfile.ZipFile, filename: str) -> str:
    """Compute SHA256 hash of a file in a ZIP archive.

    Returns hexadecimal hash string.
    """
    sha256 = hashlib.sha256()
    with zipf.open(filename) as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _normalize_media_path(path: str) -> str:
    """Normalize media path by stripping angle brackets and media/ prefix.

    Args:
        path: Raw path string from markdown/HTML

    Returns:
        Normalized path without angle brackets or media/ prefix
    """
    path = path.strip("<>")
    if path.startswith("media/"):
        path = path[6:]
    return path


def update_media_references(text: str, rename_map: dict[str, str]) -> str:
    """Update media file references in text based on rename map.

    Args:
        text: The text containing media references
        rename_map: Dictionary mapping original filenames to renamed filenames

    Returns:
        Updated text with renamed media references
    """
    if not rename_map:
        return text

    def replace_media_ref(match, add_prefix=True):
        """Replace media reference if it's in the rename map."""
        original_path = match.group(1)
        normalized_path = _normalize_media_path(original_path)

        if normalized_path in rename_map:
            new_path = rename_map[normalized_path]
            if add_prefix:
                new_path = f"media/{new_path}"
            return match.group(0).replace(original_path, new_path)
        return match.group(0)

    # Update markdown images: ![alt](media/filename.png)
    text = re.sub(
        MARKDOWN_IMAGE_PATTERN, lambda m: replace_media_ref(m, add_prefix=True), text
    )

    # Update Anki sound tags: [sound:audio.mp3] (no prefix)
    text = re.sub(
        ANKI_SOUND_PATTERN, lambda m: replace_media_ref(m, add_prefix=False), text
    )

    # Update HTML img tags: <img src="media/file.jpg">
    text = re.sub(
        HTML_IMG_PATTERN, lambda m: replace_media_ref(m, add_prefix=True), text
    )

    return text


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
            media_files.add(path)

    return media_files


def serialize_collection_to_json(
    collection_dir: Path,
    output_file: Path,
    include_ids: bool = True,
    include_media: bool = False,
) -> dict:
    """Serialize entire collection to JSON format.

    Args:
        collection_dir: Path to the collection directory
        output_file: Path where JSON file will be written
        include_ids: If False, exclude note_id and deck_id from serialized output
        include_media: If True, create ZIP with JSON and media files

    Returns:
        Dictionary containing the serialized data
    """
    # Read collection config
    marker_path = collection_dir / MARKER_FILE
    if not marker_path.exists():
        raise ValueError(f"Not a AnkiOps collection: {collection_dir}")

    # Parse .ankiops config file to get media_dir
    config_content = marker_path.read_text()
    media_dir_path = None

    for line in config_content.split("\n"):
        line = line.strip()
        if line.startswith("media_dir ="):
            media_dir_path = line.split("=", 1)[1].strip()
            break

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

        deck_data = {
            "name": md_file.stem.replace("__", "::"),  # Restore :: from __
            "notes": [],
        }

        # Only include deck_id if requested
        if include_ids:
            deck_data["deck_id"] = str(deck_id) if deck_id else None

        line_number = 1  # Track for error reporting
        for block_text in note_blocks_raw:
            block_text = block_text.strip()
            if not block_text:
                continue

            try:
                parsed = Note.from_block(block_text)

                # Convert to JSON-friendly format (note_type inferred from fields)
                note_data = {"fields": parsed.fields}

                # Only include note_id if requested
                if include_ids:
                    note_data["note_id"] = (
                        str(parsed.note_id) if parsed.note_id else None
                    )

                # Extract media references from all fields
                if include_media:
                    for field_content in parsed.fields.values():
                        all_media_files.update(extract_media_references(field_content))

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

    if include_media and all_media_files:
        # Create ZIP with JSON and media files
        # Ensure output file has .zip extension
        if not output_file.suffix == ".zip":
            output_file = output_file.with_suffix(".zip")

        logger.debug(f"Found {len(all_media_files)} media file(s) to include")

        # Get media directory from config
        if not media_dir_path:
            logger.warning("media_dir not found in config, creating JSON without media")
            include_media = False
        elif not Path(media_dir_path).exists():
            msg = f"Media directory not found at {media_dir_path}"
            logger.warning(f"{msg}, creating JSON without media")
            include_media = False
        else:
            media_base_dir = Path(media_dir_path)
            with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add JSON to ZIP
                json_content = json.dumps(serialized_data, indent=2, ensure_ascii=False)
                zipf.writestr("collection.json", json_content)

                # Add media files to ZIP
                copied_count = 0
                for media_file in all_media_files:
                    # Media file path preserves structure (e.g., AnkiOpsMedia/image.png)
                    media_path = media_base_dir / media_file
                    if media_path.exists() and media_path.is_file():
                        zipf.write(media_path, f"media/{media_file}")
                        copied_count += 1
                    else:
                        logger.warning(f"Media file not found: {media_path}")

                total_media = len(all_media_files)
                logger.debug(f"Included {copied_count}/{total_media} media files")

            logger.info(
                f"Serialized {total_decks} deck(s), {total_notes} note(s), "
                f"{copied_count} media file(s) to {output_file}"
            )

    if not (include_media and all_media_files):
        # Write JSON file only (either media not requested or no media found)
        if include_media and not all_media_files:
            logger.debug("No media files found, creating JSON only")

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


def deserialize_collection_from_json(json_file: Path, overwrite: bool = False) -> None:
    """Deserialize collection from JSON or ZIP format.

    In development mode (pyproject.toml with name="ankiops" in cwd),
    unpacks to ./collection. Otherwise, unpacks to the current working directory.

    Note: This only extracts markdown and media files. Run 'ankiops init' after
    deserializing to set up the .ankiops config file with your profile settings.

    Args:
        json_file: Path to JSON or ZIP file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
    """
    total_media = 0
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

    # Check if input is a ZIP file
    if json_file.suffix == ".zip":
        logger.debug("Detected ZIP file, extracting...")
        with zipfile.ZipFile(json_file, "r") as zipf:
            # Load JSON from ZIP
            with zipf.open("collection.json") as f:
                data = json.load(f)

            # Extract media files if present
            media_files = [
                name for name in zipf.namelist() if name.startswith("media/")
            ]
            # Track media file renames for updating references
            media_rename_map = {}

            if media_files:
                # Create media directory
                media_dir = root_dir / "media" / "AnkiOpsMedia"
                media_dir.mkdir(parents=True, exist_ok=True)

                # Extract media files with conflict handling
                extracted_count = 0
                skipped_count = 0
                renamed_count = 0

                for media_file in media_files:
                    # Remove 'media/' prefix to get just the filename
                    filename = media_file.replace("media/", "")
                    if not filename:  # Skip if empty (directory entry)
                        continue

                    target = media_dir / filename

                    # Always use conflict resolution for media files
                    # (overwrite flag only applies to markdown/config files)
                    if target.exists():
                        # Compute hash of existing file
                        existing_hash = compute_file_hash(target)
                        # Compute hash of file in ZIP
                        new_hash = compute_zipfile_hash(zipf, media_file)

                        if existing_hash == new_hash:
                            # Same file, skip extraction
                            logger.debug(f"Skipping {filename} (identical file exists)")
                            skipped_count += 1
                            continue
                        else:
                            # Different file with same name, find unique name
                            base_name = target.stem
                            extension = target.suffix
                            counter = 1
                            while True:
                                new_filename = f"{base_name}_{counter}{extension}"
                                new_target = media_dir / new_filename
                                if not new_target.exists():
                                    target = new_target
                                    # Track rename for updating references
                                    media_rename_map[filename] = new_filename
                                    logger.info(
                                        f"Renaming {filename} → {new_filename} (conflict)"
                                    )
                                    renamed_count += 1
                                    break
                                counter += 1

                    # Extract the file
                    source = zipf.open(media_file)
                    with target.open("wb") as f:
                        shutil.copyfileobj(source, f)
                    source.close()
                    extracted_count += 1

                total_media = extracted_count
                summary_parts = [f"Extracted {extracted_count} media file(s)"]
                if skipped_count > 0:
                    summary_parts.append(f"skipped {skipped_count} duplicate(s)")
                if renamed_count > 0:
                    summary_parts.append(f"renamed {renamed_count} conflict(s)")
                logger.debug(", ".join(summary_parts))
    else:
        # Load JSON data directly
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # No media files in JSON-only format
        media_rename_map = {}

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

        # Add deck_id if present
        if deck_id:
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

            # Add note_id if present
            if note_id:
                lines.append(f"<!-- note_id: {note_id} -->")

            # Get field mappings for this note type
            note_config = NOTE_CONFIG.get(note_type)
            if not note_config:
                logger.warning(
                    f"Unknown note type '{note_type}' in deck '{deck_name}', skipping note"
                )
                continue

            # Format fields according to note type configuration
            for field_name, prefix in note_config:
                field_content = fields.get(field_name)
                if field_content:
                    # Update media references if files were renamed
                    field_content = update_media_references(
                        field_content, media_rename_map
                    )
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
            output_path.write_text(content)
            logger.info(f"  Created {clickable_path(output_path)} ({len(notes)} notes)")
        else:
            logger.debug(
                f"Skipped {clickable_path(output_path)} (already exists, use --overwrite to replace)"
            )

        total_decks += 1
        total_notes += len(notes)

    media_part = f", {total_media} media file(s)" if total_media else ""
    logger.info(
        f"Deserialized {total_decks} deck(s), {total_notes} note(s){media_part}"
        f" to {root_dir}"
    )

    # Check if .ankiops marker file exists
    marker_path = root_dir / MARKER_FILE
    if not marker_path.exists():
        logger.info(
            "Run 'ankiops init' to set up this collection with your Anki profile."
        )

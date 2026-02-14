"""Package and unpackage AnkiOps collections to/from JSON format."""

import hashlib
import json
import logging
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ankiops.config import MARKER_FILE
from ankiops.markdown_helpers import (
    extract_deck_id,
    parse_note_block,
)

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_zipfile_hash(zipf: zipfile.ZipFile, filename: str) -> str:
    """Compute SHA256 hash of a file in a ZIP archive.

    Args:
        zipf: ZipFile object
        filename: Name of file within the ZIP

    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()
    with zipf.open(filename) as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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

    updated_text = text

    # Update markdown images: ![alt](filename.png)
    def replace_md_image(match):
        path = match.group(1).strip("<>")
        original_path = path
        if path.startswith("media/"):
            path = path[6:]
        if path in rename_map:
            new_path = rename_map[path]
            return match.group(0).replace(original_path, f"media/{new_path}")
        return match.group(0)

    updated_text = re.sub(
        r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?", replace_md_image, updated_text
    )

    # Update Anki sound tags: [sound:audio.mp3]
    def replace_sound(match):
        path = match.group(1).strip("<>")
        original_path = path
        if path.startswith("media/"):
            path = path[6:]
        if path in rename_map:
            new_path = rename_map[path]
            return match.group(0).replace(original_path, new_path)
        return match.group(0)

    updated_text = re.sub(r"\[sound:([^\]]+)\]", replace_sound, updated_text)

    # Update HTML img tags: <img src="file.jpg">
    def replace_html_img(match):
        path = match.group(1).strip("<>")
        original_path = path
        if path.startswith("media/"):
            path = path[6:]
        if path in rename_map:
            new_path = rename_map[path]
            return match.group(0).replace(original_path, f"media/{new_path}")
        return match.group(0)

    updated_text = re.sub(
        r'<img[^>]+src=["\']([^"\']+)["\']', replace_html_img, updated_text
    )

    return updated_text


def extract_media_references(text: str) -> set[str]:
    """Extract media file references from markdown text.

    Finds:
    - Markdown images: ![alt](filename.png)
    - Anki sound: [sound:audio.mp3]
    - HTML img tags: <img src="file.jpg">

    Returns:
        Set of media file paths (preserves relative path structure)
    """
    media_files = set()

    # Markdown images: ![alt](filename.png) or ![alt](filename.png){width=500}
    for match in re.finditer(r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?", text):
        path = match.group(1)
        # Strip angle brackets if present (markdown URL syntax)
        path = path.strip("<>")
        # Remove leading 'media/' if present
        if path.startswith("media/"):
            path = path[6:]  # Remove 'media/' prefix
        media_files.add(path)

    # Anki sound tags: [sound:audio.mp3]
    for match in re.finditer(r"\[sound:([^\]]+)\]", text):
        path = match.group(1).strip("<>")
        if path.startswith("media/"):
            path = path[6:]
        media_files.add(path)

    # HTML img tags: <img src="file.jpg">
    for match in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', text):
        path = match.group(1).strip("<>")
        if path.startswith("media/"):
            path = path[6:]
        media_files.add(path)

    return media_files


def package_collection_to_json(
    collection_dir: Path,
    output_file: Path,
    include_ids: bool = True,
    include_media: bool = False,
) -> dict:
    """Package entire collection to JSON format.

    Args:
        collection_dir: Path to the collection directory
        output_file: Path where JSON file will be written
        include_ids: If False, exclude note_id and deck_id from package
        include_media: If True, create ZIP with JSON and media files

    Returns:
        Dictionary containing the packaged data
    """
    # Read collection config
    marker_path = collection_dir / MARKER_FILE
    if not marker_path.exists():
        raise ValueError(f"Not a AnkiOps collection: {collection_dir}")

    # Parse .ankiops config file
    config_content = marker_path.read_text()
    profile = None
    media_dir_path = None
    auto_commit = True

    for line in config_content.split("\n"):
        line = line.strip()
        if line.startswith("profile ="):
            profile = line.split("=", 1)[1].strip()
        elif line.startswith("media_dir ="):
            media_dir_path = line.split("=", 1)[1].strip()
        elif line.startswith("auto_commit ="):
            auto_commit = line.split("=", 1)[1].strip().lower() == "true"

    # Build JSON structure
    package_data = {
        "collection": {
            "profile": profile,
            "auto_commit": auto_commit,
            "packaged_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    # Track all media files referenced in notes
    all_media_files = set()

    # Process all markdown files in collection
    md_files = sorted(collection_dir.glob("*.md"))
    logger.debug(f"Found {len(md_files)} deck file(s) to package")

    for md_file in md_files:
        logger.debug(f"Processing {md_file.name}...")
        content = md_file.read_text()

        # Extract deck_id and remaining content
        deck_id, remaining_content = extract_deck_id(content)

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
                parsed = parse_note_block(block_text)

                # Convert to JSON-friendly format
                note_data = {
                    "note_type": parsed.note_type,
                    "fields": parsed.fields,
                }

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
                # Continue processing other notes
                line_number += block_text.count("\n") + 3

        if deck_data["notes"]:
            package_data["decks"].append(deck_data)

    total_notes = sum(len(deck["notes"]) for deck in package_data["decks"])
    total_decks = len(package_data["decks"])

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
                json_content = json.dumps(package_data, indent=2, ensure_ascii=False)
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
                f"Packaged {total_decks} deck(s), {total_notes} note(s), "
                f"{copied_count} media file(s) to {output_file}"
            )

    if not (include_media and all_media_files):
        # Write JSON file only (either media not requested or no media found)
        if include_media and not all_media_files:
            logger.debug("No media files found, creating JSON only")

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(package_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Packaged {total_decks} deck(s), {total_notes} note(s) to {output_file}"
        )

    return package_data


def unpackage_collection_from_json(
    json_file: Path, collection_dir: Path, overwrite: bool = False
) -> None:
    """Unpackage collection from JSON or ZIP format.

    Args:
        json_file: Path to JSON or ZIP file to unpackage
        collection_dir: Path to collection directory (will be created if doesn't exist)
        overwrite: If True, overwrite existing collection; if False, merge with existing
    """
    total_media = 0

    # Check if input is a ZIP file
    if json_file.suffix == ".zip":
        logger.debug("Detected ZIP file, extracting...")
        with zipfile.ZipFile(json_file, "r") as zipf:
            # Load JSON from ZIP
            with zipf.open("collection.json") as f:
                data = json.load(f)

            # Create collection directory if it doesn't exist
            collection_dir.mkdir(parents=True, exist_ok=True)

            # Extract media files if present
            media_files = [
                name for name in zipf.namelist() if name.startswith("media/")
            ]
            # Track media file renames for updating references
            media_rename_map = {}

            if media_files:
                # Create media directory
                media_dir = collection_dir / "media" / "AnkiOpsMedia"
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

        # Create collection directory if it doesn't exist
        collection_dir.mkdir(parents=True, exist_ok=True)

        # No media files in JSON-only format
        media_rename_map = {}

    # Create collection directory if it doesn't exist
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Write .ankiops config file
    marker_path = collection_dir / MARKER_FILE
    if overwrite or not marker_path.exists():
        profile = data["collection"].get("profile", "default")
        auto_commit = data["collection"].get("auto_commit", True)

        config_content = f"""# AnkiOps collection — do not delete this file.

[ankiops]
profile = {profile}
auto_commit = {str(auto_commit).lower()}

"""
        marker_path.write_text(config_content)
        logger.debug(f"Created collection config for profile '{profile}'")

    # Process each deck
    from ankiops.config import NOTE_TYPES

    total_decks = 0
    total_notes = 0

    for deck in data["decks"]:
        deck_name = deck["name"]
        deck_id = deck.get("deck_id")
        notes = deck["notes"]

        # Sanitize filename (replace :: with __)
        filename = deck_name.replace("::", "__") + ".md"
        output_path = collection_dir / filename

        # Build markdown content
        lines = []

        # Add deck_id if present
        if deck_id:
            lines.append(f"<!-- deck_id: {deck_id} -->")

        # Process each note
        for note in notes:
            note_id = note.get("note_id")
            note_type = note["note_type"]
            fields = note["fields"]

            # Add note_id if present
            if note_id:
                lines.append(f"<!-- note_id: {note_id} -->")

            # Get field mappings for this note type
            note_config = NOTE_TYPES.get(note_type)
            if not note_config:
                logger.warning(
                    f"Unknown note type '{note_type}' in deck '{deck_name}', skipping note"
                )
                continue

            # Format fields according to note type configuration
            for field_name, prefix, mandatory in note_config["field_mappings"]:
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
            logger.info(f"  Created {filename} ({len(notes)} notes)")
        else:
            logger.debug(
                f"Skipped {filename} (already exists, use --overwrite to replace)"
            )

        total_decks += 1
        total_notes += len(notes)

    media_part = f", {total_media} media file(s)" if total_media else ""
    logger.info(
        f"Unpackaged {total_decks} deck(s), {total_notes} note(s){media_part}"
        f" to {collection_dir}"
    )

"""Media synchronization logic for AnkiOps.

Handles:
- Proactive hashing (renaming files to include content hash)
- Bidirectional sync (Import: Local -> Anki, Export: Anki -> Local)
- Updating markdown references when files are renamed
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.log import clickable_path
from ankiops.models import (
    Change,
    ChangeType,
    MediaSyncResult,
    NoteSyncResult,
)

logger = logging.getLogger(__name__)

HASH_SUFFIX_PATTERN = re.compile(r"_([a-f0-9]{8})\.[^.]+$")

ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
MARKDOWN_IMAGE_PATTERN = r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?"
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


def _calculate_hash(file_path: Path) -> str:
    """Calculate 8-char BLAKE2b hash of a file."""
    # usage of blake2b is faster than sha256 on 64-bit systems
    h = hashlib.blake2b(digest_size=4)
    with open(file_path, "rb") as f:
        # 64KB chunks for optimal I/O
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_hashed_filename(file_path: Path, content_hash: str) -> str:
    """Return filename with hash suffix (e.g. image_a1b2c3.png)."""
    stem = file_path.stem
    suffix = file_path.suffix

    # Check if already hashed
    match = HASH_SUFFIX_PATTERN.search(file_path.name)
    if match:
        existing_hash = match.group(1)
        if existing_hash == content_hash:
            return file_path.name  # Already correct
        # Strip old hash
        stem = stem[: -len(existing_hash) - 1]

    return f"{stem}_{content_hash}{suffix}"


def update_markdown_media_references(
    collection_dir: Path, rename_map: dict[str, str]
) -> int:
    """Update all markdown files in collection to use new filenames.

    Args:
        collection_dir: Root of standard markdown collection
        rename_map: Dict mapping {original_relative_path_str: new_relative_path_str}
                   Note: paths should use forward slashes for markdown compatibility.

    Returns:
        Number of markdown files updated.
    """
    if not rename_map:
        return 0

    updated_files = 0

    # Regex to find potential media links:
    # 1. Markdown image: ![alt](path) -> group 2
    # 2. HTML img: src="path" -> group 3
    # 3. Anki sound: [sound:path] -> group 4
    # We capture the path to check against rename_map
    pattern = re.compile(r'(!\[.*?\]\((.+?)\)|src=["\'](.+?)["\']|\[sound:(.+?)\])')

    def replace_callback(match: re.Match) -> str:
        full_match = match.group(1)
        # Extract path from whichever group matched
        path = match.group(2) or match.group(3) or match.group(4)

        # Normalize path for lookup (handle windows backslashes if any)
        # Also strip angle brackets if present (markdown can use <path>)
        lookup_path = path.strip("<>").replace("\\", "/")

        if lookup_path in rename_map:
            new_path = rename_map[lookup_path]

            # Always wrap the new path in angle brackets for consistency and robustness
            new_path = f"<{new_path}>"

            # Replace the old path with new path in the full match string
            return full_match.replace(path, new_path)

        return full_match

    md_files = list(collection_dir.glob("*.md"))
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")

        # Perform replacement in one pass
        new_content = pattern.sub(replace_callback, content)

        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            logger.debug(f"Updated media references in {clickable_path(md_file)}")
            updated_files += 1

    return updated_files


def apply_hashing(
    collection_dir: Path,
    referenced_files: set[str] | None = None,
    result: "NoteSyncResult" | None = None,
) -> dict[str, str]:
    """Rename local media files to include content hash and update markdown refs.

    Recursive scan of LOCAL_MEDIA_DIR.

    Args:
        collection_dir: Root of the collection
        referenced_files: Optional set of filenames that are actually used.
                         If provided, only files in this set (or starting with _)
                         will be hashed.
        result: Optional NoteSyncResult to update.

    Returns:
        Number of media files hashed.
    """
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return {}

    rename_map = {}
    hashed_count = 0

    for file_path in media_root.glob("*"):
        if (
            not file_path.is_file()
            or file_path.name.startswith(".")
            or file_path.name.startswith("_")
        ):
            continue

        if referenced_files is not None and file_path.name not in referenced_files:
            continue

        try:
            content_hash = _calculate_hash(file_path)
            new_name = _get_hashed_filename(file_path, content_hash)

            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)
                if new_path.exists():
                    if _calculate_hash(new_path) == content_hash:
                        file_path.unlink()
                    else:
                        continue
                else:
                    file_path.rename(new_path)

                hashed_count += 1
                rename_map[f"{LOCAL_MEDIA_DIR}/{file_path.name}"] = (
                    f"{LOCAL_MEDIA_DIR}/{new_name}"
                )
                rename_map[file_path.name] = f"{LOCAL_MEDIA_DIR}/{new_name}"
                if result:
                    result.changes.append(
                        Change(
                            ChangeType.HASH,
                            file_path.name,
                            file_path.name,
                            {"new_name": new_name},
                        )
                    )

        except Exception as e:
            logger.warning(
                f"Failed to hash media file {clickable_path(file_path)}: {e}"
            )
            if result:
                result.errors.append(str(e))

    if rename_map:
        count = update_markdown_media_references(collection_dir, rename_map)
        logger.debug(f"Updated {count} markdown files with {len(rename_map)} renames")

    return rename_map


def sync_to_anki(collection_dir: Path, anki_media_dir: Path) -> MediaSyncResult:
    """Sync local media to Anki collection.media."""
    result = MediaSyncResult()
    target_root = Path(anki_media_dir).resolve()
    media_root = (collection_dir / LOCAL_MEDIA_DIR).resolve()

    if media_root == target_root:
        logger.error(
            f"Local media directory ({media_root}) aliases Anki media directory."
        )
        raise ValueError(
            f"Local media directory ({media_root}) aliases Anki media directory. "
            "Syncing would rename all files in Anki. Aborting."
        )

    # 1. Identify all referenced media files BEFORE hashing
    referenced_files = set()
    for md_file in collection_dir.glob("*.md"):
        referenced_files.update(
            extract_media_references(md_file.read_text(encoding="utf-8"))
        )

    # 2. Apply hashing (only for referenced files)
    # create a dummy note sync result to collect changes from hashing
    note_result = NoteSyncResult(deck_name="media", file_path=None)
    rename_map = apply_hashing(
        collection_dir, referenced_files=referenced_files, result=note_result
    )
    result.changes.extend(note_result.changes)
    result.errors.extend(note_result.errors)

    # Refresh referenced files in case some were renamed by hashing
    referenced_files = set()
    for md_file in collection_dir.glob("*.md"):
        referenced_files.update(
            extract_media_references(md_file.read_text(encoding="utf-8"))
        )

    media_root = (collection_dir / LOCAL_MEDIA_DIR).resolve()
    if not media_root.exists():
        return result

    # Create inverse map to find original names after hashing
    inv_rename_map = {v.split("/")[-1]: k.split("/")[-1] for k, v in rename_map.items()}

    for file_path in media_root.glob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        filename = file_path.name

        if filename in referenced_files or filename.startswith("_"):
            target_path = target_root / filename
            if not target_path.exists():
                shutil.copy2(file_path, target_path)
                orig_name = inv_rename_map.get(filename, filename)
                result.changes.append(Change(ChangeType.SYNC, orig_name, filename))

        if filename not in referenced_files and not filename.startswith("_"):
            try:
                file_path.unlink()
                result.changes.append(Change(ChangeType.DELETE, filename, filename))
            except Exception as e:
                result.errors.append(str(e))

    return result


def sync_from_anki(
    collection_dir: Path,
    anki_media_dir: Path,
    referenced_files: set[str],
) -> MediaSyncResult:
    """Sync referenced media from Anki to local media folder."""
    result = MediaSyncResult()
    media_root = collection_dir / LOCAL_MEDIA_DIR
    media_root.mkdir(parents=True, exist_ok=True)
    target_root = Path(anki_media_dir)

    for filename in referenced_files:
        source_path = target_root / filename
        target_path = media_root / filename

        if not source_path.exists():
            logger.warning(
                f"Referenced media not found in Anki: {clickable_path(source_path, filename)}"
            )
            result.changes.append(Change(ChangeType.SKIP, filename, filename))
            continue

        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            logger.debug(f"Synced {clickable_path(source_path, filename)} from Anki")
            result.changes.append(Change(ChangeType.SYNC, filename, filename))
        else:
            result.changes.append(Change(ChangeType.SKIP, filename, filename))

    return result

"""Media synchronization logic for AnkiOps.

Handles:
- Proactive hashing (renaming files to include content hash)
- Bidirectional sync (Import: Local -> Anki, Export: Anki -> Local)
- Updating markdown references when files are renamed
"""

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)

HASH_SUFFIX_PATTERN = re.compile(r"_([a-f0-9]{8})\.[^.]+$")

ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
MARKDOWN_IMAGE_PATTERN = r"!\[.*?\]\(([^)]+?)\)(?:\{[^}]*\})?"
HTML_IMG_PATTERN = r'<img[^>]+src=["\']([^"\']+)["\']'


@dataclass
class MediaSyncSummary:
    """Summary of media synchronization operations."""

    synced: int = 0
    removed: int = 0
    hashed: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for compatibility with format_changes."""
        return {
            "synced": self.synced,
            "removed": self.removed,
            "hashed": self.hashed,
            "total": self.total,
        }


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
) -> None:
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
    summary: MediaSyncSummary | None = None,
) -> int:
    """Rename local media files to include content hash and update markdown refs.

    Recursive scan of LOCAL_MEDIA_DIR.

    Args:
        collection_dir: Root of the collection
        referenced_files: Optional set of filenames that are actually used.
                         If provided, only files in this set (or starting with _)
                         will be hashed.
        summary: Optional MediaSyncSummary to update.

    Returns:
        Number of media files hashed.
    """
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return 0

    rename_map = {}  # old_rel_path -> new_rel_path
    hashed_count = 0

    # Walk all files (non-recursive)
    for file_path in media_root.glob("*"):
        if (
            not file_path.is_file()
            or file_path.name.startswith(".")
            or file_path.name.startswith("_")
        ):
            continue

        # If referenced_files provided, skip hashing unreferenced ones
        if referenced_files is not None and file_path.name not in referenced_files:
            continue

        try:
            content_hash = _calculate_hash(file_path)
            new_name = _get_hashed_filename(file_path, content_hash)

            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)

                # Handle collision if target exists
                if new_path.exists():
                    # If content matches, we can delete the old unhashed file
                    if _calculate_hash(new_path) == content_hash:
                        file_path.unlink()
                        logger.debug(
                            f"Replaced {clickable_path(file_path)} with existing "
                            f"{clickable_path(new_path)}"
                        )
                    else:
                        logger.warning(
                            f"Target {clickable_path(new_path)} exists but content "
                            f"differs? Skipping rename of {clickable_path(file_path)}"
                        )
                        continue
                else:
                    file_path.rename(new_path)
                    logger.debug(
                        f"Hashed: {clickable_path(file_path)} -> "
                        f"{clickable_path(new_path)}"
                    )

                hashed_count += 1

                # Record relative paths for markdown update
                old_rel = f"{LOCAL_MEDIA_DIR}/{file_path.name}"
                new_rel = f"{LOCAL_MEDIA_DIR}/{new_name}"
                rename_map[old_rel] = new_rel
                # Also map the bare filename itself to the new path (with media/ prefix)
                # This handles cases where markdown refs don't utilize the prefix (e.g. ![alt](img.png))
                rename_map[file_path.name] = new_rel

        except Exception as e:
            logger.warning(
                f"Failed to hash media file {clickable_path(file_path)}: {e}"
            )

    if rename_map:
        update_markdown_media_references(collection_dir, rename_map)

    if summary:
        summary.hashed += hashed_count

    return hashed_count


def sync_to_anki(collection_dir: Path, anki_media_dir: Path) -> MediaSyncSummary:
    """Sync local media to Anki collection.media.

    1. Extract referenced files.
    2. Apply hashing (only to referenced files).
    3. Copy all files to Anki's media directory.
    4. Remove unreferenced files from local.

    Returns:
        MediaSyncSummary with counts of synced, removed, and hashed files.
    """
    summary = MediaSyncSummary()
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
    apply_hashing(collection_dir, referenced_files=referenced_files, summary=summary)

    # Refresh referenced files in case some were renamed by hashing
    referenced_files = set()
    for md_file in collection_dir.glob("*.md"):
        referenced_files.update(
            extract_media_references(md_file.read_text(encoding="utf-8"))
        )

    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return summary

    target_root = Path(anki_media_dir)
    local_files = list(media_root.glob("*"))

    # 3. Iterate over local files to Sync or Delete
    for file_path in local_files:
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        filename = file_path.name

        # If referenced OR underscore file: Sync to Anki
        if filename in referenced_files or filename.startswith("_"):
            target_path = target_root / filename
            if not target_path.exists():
                shutil.copy2(file_path, target_path)
                logger.debug(f"Synced {clickable_path(file_path)} to Anki")
                summary.synced += 1

        # If NOT referenced: Check for cleanup
        if filename not in referenced_files:
            # Keep special files starting with _ (Anki templates/static assets)
            if filename.startswith("_"):
                continue

            # otherwise delete
            try:
                file_path.unlink()
                logger.debug(
                    f"Removed unreferenced media file: {clickable_path(file_path)}"
                )
                summary.removed += 1
            except Exception as e:
                logger.warning(
                    f"Failed to remove unreferenced file {clickable_path(file_path)}: {e}"
                )

    # Final total: files remaining in local media folder
    summary.total = len(
        [f for f in media_root.glob("*") if f.is_file() and not f.name.startswith(".")]
    )

    return summary


def sync_from_anki(
    collection_dir: Path,
    anki_media_dir: Path,
    referenced_files: set[str],
) -> MediaSyncSummary:
    """Sync referenced media from Anki to local media folder.

    Only copies files that are referenced in the markdown.

    Returns:
        MediaSyncSummary with counts.
    """
    summary = MediaSyncSummary(total=len(referenced_files))
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
            continue

        if not target_path.exists():
            shutil.copy2(source_path, target_path)
            logger.debug(f"Synced {clickable_path(source_path, filename)} from Anki")
            summary.synced += 1

    return summary

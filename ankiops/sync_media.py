"""Use Case: Synchronize Media Files with Anki."""

import logging
import re
import shutil
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path
from ankiops.models import Change, ChangeType, MediaSyncResult

from urllib.parse import unquote

logger = logging.getLogger(__name__)

HASH_SUFFIX_PATTERN = re.compile(r"_([a-f0-9]{8})\.[^.]+$")
ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
MARKDOWN_IMAGE_PATTERN = r"!\[.*?\]\((?:<(.+?)>|([^()]+(?:\([^()]*\)[^()]*)*))\)(?:\{[^}]*\})?"
HTML_IMG_PATTERN = r'<img[^>]+src=["\']([^"\']+)["\']'


def _normalize_media_path(path: str) -> str:
    path = path.strip("<>")
    if path.startswith(f"{LOCAL_MEDIA_DIR}/"):
        path = path[len(LOCAL_MEDIA_DIR) + 1 :]
    return path


def _extract_media_references(text: str) -> set[str]:
    media_files = set()
    for pattern in [MARKDOWN_IMAGE_PATTERN, ANKI_SOUND_PATTERN, HTML_IMG_PATTERN]:
        for match in re.finditer(pattern, text):
            # For Markdown pattern, path is in group 1 OR group 2. For others, it's group 1.
            # Using len(match.groups()) > 1 to safely check if it's the Markdown pattern.
            if len(match.groups()) > 1 and match.re.pattern == MARKDOWN_IMAGE_PATTERN:
                raw_path = match.group(1) or match.group(2)
            else:
                raw_path = match.group(1)
            
            if not raw_path:
                continue

            decoded_path = unquote(raw_path)
            path = _normalize_media_path(decoded_path)
            if path and not path.startswith(("http://", "https://")):
                media_files.add(path)
    return media_files


def _get_hashed_name(file_path: Path, digest: str) -> str:
    stem = file_path.stem
    suffix = file_path.suffix
    match = HASH_SUFFIX_PATTERN.search(file_path.name)
    if match:
        existing = match.group(1)
        if existing == digest:
            return file_path.name
        stem = stem[: -len(existing) - 1]
    return f"{stem}_{digest}{suffix}"


def hash_and_update_references(
    fs_port: FileSystemAdapter,
    collection_dir: Path,
    result: MediaSyncResult,
) -> set[str]:
    """Hash files, rename them locally, and update markdown."""
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return set()

    # 1. Find directly referenced files first
    referenced = set()
    for md_file in fs_port.find_markdown_files(collection_dir):
        raw_content = md_file.read_text(encoding="utf-8")
        referenced.update(_extract_media_references(raw_content))

    rename_map = {}

    # 2. Hash referenced (or already hashed) files
    for file_path in media_root.glob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.name not in referenced and not file_path.name.startswith("_"):
            continue

        try:
            digest = fs_port.calculate_blake3(file_path)
            new_name = _get_hashed_name(file_path, digest)

            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)
                if new_path.exists():
                    if fs_port.calculate_blake3(new_path) == digest:
                        file_path.unlink()
                    else:
                        continue
                else:
                    file_path.rename(new_path)

                rename_map[f"{LOCAL_MEDIA_DIR}/{file_path.name}"] = (
                    f"{LOCAL_MEDIA_DIR}/{new_name}"
                )
                rename_map[file_path.name] = f"{LOCAL_MEDIA_DIR}/{new_name}"
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
            result.errors.append(str(e))

    # 3. Update Markdown files in bulk
    if rename_map:
        count = fs_port.update_media_references(collection_dir, rename_map)
        logger.debug(f"Updated {count} markdown files with {len(rename_map)} renames")

    # 4. Re-collect now-correctly-hashed active references
    final_referenced = set()
    for md_file in fs_port.find_markdown_files(collection_dir):
        raw_content = md_file.read_text(encoding="utf-8")
        final_referenced.update(_extract_media_references(raw_content))

    return final_referenced


def sync_media_to_anki(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    collection_dir: Path,
) -> MediaSyncResult:
    """Push local media to Anki."""
    result = MediaSyncResult()
    media_root = (collection_dir / LOCAL_MEDIA_DIR).resolve()

    anki_media_dir = anki_port.get_media_dir().resolve()
    if media_root == anki_media_dir:
        raise ValueError(
            f"Local media directory ({media_root}) is the same as Anki's media "
            "directory. Syncing would rename files inside Anki directly. Aborting."
        )

    active_refs = hash_and_update_references(fs_port, collection_dir, result)
    if not media_root.exists():
        return result

    for file_path in media_root.glob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        name = file_path.name
        if name in active_refs or name.startswith("_"):
            # Push safely via Port
            anki_port.push_media(file_path, name)
            result.changes.append(Change(ChangeType.SYNC, name, name))
            logger.debug(f"  Synced {clickable_path(file_path)}")
        elif name not in active_refs and not name.startswith("_"):
            try:
                # Store the path before unlinking it so clickable_path can see it exists
                p = file_path
                file_path.unlink()
                result.changes.append(Change(ChangeType.DELETE, name, name))
                logger.debug(f"  Deleted orphan {clickable_path(p)}")
            except Exception as e:
                result.errors.append(str(e))

    return result


def sync_media_from_anki(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    collection_dir: Path,
) -> MediaSyncResult:
    """Pull missing media from Anki."""
    result = MediaSyncResult()
    media_root = collection_dir / LOCAL_MEDIA_DIR
    media_root.mkdir(parents=True, exist_ok=True)

    referenced = set()
    for md_file in fs_port.find_markdown_files(collection_dir):
        raw_content = md_file.read_text(encoding="utf-8")
        referenced.update(_extract_media_references(raw_content))

    for name in referenced:
        target = media_root / name
        if not target.exists():
            success = anki_port.pull_media(name, target)
            if success:
                result.changes.append(Change(ChangeType.SYNC, name, name))
                logger.debug(f"  Pulled {clickable_path(target)} from Anki")
            else:
                logger.warning(f"Media {name} missing in Anki")
                result.changes.append(Change(ChangeType.SKIP, name, name))
        else:
            result.changes.append(Change(ChangeType.SKIP, name, name))

    return result

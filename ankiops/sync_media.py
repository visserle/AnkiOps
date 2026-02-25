"""Use Case: Synchronize Media Files with Anki."""

import logging
import re
from pathlib import Path
from urllib.parse import unquote

from ankiops.anki import AnkiAdapter
from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path
from ankiops.models import Change, ChangeType, MediaSyncResult

logger = logging.getLogger(__name__)

HASH_SUFFIX_PATTERN = re.compile(r"_([a-f0-9]{8})\.[^.]+$")
ANKI_SOUND_PATTERN = r"\[sound:([^\]]+)\]"
MARKDOWN_IMAGE_PATTERN = (
    r"!\[.*?\]\((?:<(.+?)>|([^()]+(?:\([^()]*\)[^()]*)*))\)(?:\{[^}]*\})?"
)
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


def _markdown_cache_key(collection_dir: Path, md_file: Path) -> str:
    return str(md_file.relative_to(collection_dir))


def _collect_referenced_media(
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
) -> set[str]:
    md_files = fs_port.find_markdown_files(collection_dir)
    md_keys = [_markdown_cache_key(collection_dir, md_file) for md_file in md_files]
    pruned = db_port.prune_markdown_media_cache(md_keys)
    if pruned:
        logger.debug(f"Pruned media cache for {pruned} stale markdown path(s)")
    cached = db_port.get_markdown_media_cache_bulk(md_keys)

    referenced: set[str] = set()
    updates: list[tuple[str, int, int, set[str]]] = []
    cache_hits = 0
    cache_misses = 0

    for md_file, md_key in zip(md_files, md_keys):
        stat = md_file.stat()
        cached_entry = cached.get(md_key)
        if (
            cached_entry
            and cached_entry[0] == stat.st_mtime_ns
            and cached_entry[1] == stat.st_size
        ):
            referenced.update(cached_entry[2])
            cache_hits += 1
            continue

        cache_misses += 1
        raw_content = md_file.read_text(encoding="utf-8")
        refs = _extract_media_references(raw_content)
        referenced.update(refs)
        updates.append((md_key, stat.st_mtime_ns, stat.st_size, refs))

    if updates:
        db_port.set_markdown_media_cache_bulk(updates)
    logger.debug(
        f"Media refs cache: {cache_hits} hits, {cache_misses} misses "
        f"across {len(md_files)} markdown files"
    )
    return referenced


def hash_and_update_references(
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    result: MediaSyncResult,
) -> tuple[set[str], dict[str, str]]:
    """Hash files, rename them locally, and update markdown."""
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return set(), {}

    # 1. Find directly referenced files first
    referenced = _collect_referenced_media(fs_port, db_port, collection_dir)

    rename_map: dict[str, str] = {}
    digest_by_name: dict[str, str] = {}
    fingerprint_updates: list[tuple[str, int, int, str, str]] = []
    fingerprint_removals: list[str] = []
    hash_cache_hits = 0
    hash_cache_misses = 0

    media_files = sorted(media_root.glob("*"), key=lambda p: p.name)
    cache_candidates = [
        file_path.name
        for file_path in media_files
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and (file_path.name in referenced or file_path.name.startswith("_"))
    ]
    cached_fingerprints = db_port.get_media_fingerprints_bulk(cache_candidates)

    # 2. Hash referenced (or already hashed) files
    for file_path in media_files:
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.name not in referenced and not file_path.name.startswith("_"):
            continue

        try:
            stat = file_path.stat()
            cached = cached_fingerprints.get(file_path.name)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                digest = cached[2]
                hash_cache_hits += 1
            else:
                digest = fs_port.calculate_blake3(file_path)
                hash_cache_misses += 1
            new_name = _get_hashed_name(file_path, digest)
            final_path = file_path

            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)
                if new_path.exists():
                    new_digest = fs_port.calculate_blake3(new_path)
                    if new_digest == digest:
                        file_path.unlink()
                        fingerprint_removals.append(file_path.name)
                    else:
                        continue
                else:
                    file_path.rename(new_path)
                    fingerprint_removals.append(file_path.name)
                final_path = new_path

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

            final_stat = final_path.stat()
            digest_by_name[final_path.name] = digest
            fingerprint_updates.append(
                (
                    final_path.name,
                    final_stat.st_mtime_ns,
                    final_stat.st_size,
                    digest,
                    _get_hashed_name(final_path, digest),
                )
            )

        except Exception as e:
            logger.warning(
                f"Failed to hash media file {clickable_path(file_path)}: {e}"
            )
            result.errors.append(str(e))

    if fingerprint_updates:
        db_port.set_media_fingerprints_bulk(fingerprint_updates)
    if fingerprint_removals:
        db_port.remove_media_fingerprints_by_names(fingerprint_removals)
        db_port.remove_media_push_state_by_names(fingerprint_removals)

    logger.debug(
        f"Media hash cache: {hash_cache_hits} hits, {hash_cache_misses} misses "
        f"across {len(cache_candidates)} candidate files"
    )

    # 3. Update Markdown files in bulk
    if rename_map:
        count = fs_port.update_media_references(collection_dir, rename_map)
        logger.debug(f"Updated {count} markdown files with {len(rename_map)} renames")

    # 4. Re-collect now-correctly-hashed active references
    final_referenced = _collect_referenced_media(fs_port, db_port, collection_dir)
    return final_referenced, digest_by_name


def sync_media_to_anki(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    collection_dir: Path,
    db_port: SQLiteDbAdapter,
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

    active_refs, digest_by_name = hash_and_update_references(
        fs_port, db_port, collection_dir, result
    )
    if not media_root.exists():
        return result

    media_files = sorted(media_root.glob("*"), key=lambda p: p.name)
    push_candidates = [
        file_path.name
        for file_path in media_files
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and (file_path.name in active_refs or file_path.name.startswith("_"))
    ]
    cached_push_state = db_port.get_media_push_state_bulk(push_candidates)
    cached_fingerprints = db_port.get_media_fingerprints_bulk(push_candidates)

    push_state_updates: list[tuple[str, str]] = []
    fingerprint_updates: list[tuple[str, int, int, str, str]] = []
    removed_names: list[str] = []
    skipped_pushes = 0

    for file_path in media_files:
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        name = file_path.name
        if name in active_refs or name.startswith("_"):
            stat = file_path.stat()
            digest = digest_by_name.get(name)
            if digest is None:
                cached = cached_fingerprints.get(name)
                if (
                    cached
                    and cached[0] == stat.st_mtime_ns
                    and cached[1] == stat.st_size
                ):
                    digest = cached[2]
                else:
                    digest = fs_port.calculate_blake3(file_path)
                    fingerprint_updates.append(
                        (
                            name,
                            stat.st_mtime_ns,
                            stat.st_size,
                            digest,
                            _get_hashed_name(file_path, digest),
                        )
                    )
                digest_by_name[name] = digest

            if (
                cached_push_state.get(name) == digest
                and (anki_media_dir / name).exists()
            ):
                skipped_pushes += 1
                continue

            # Push safely via Port
            anki_port.push_media(file_path, name)
            result.changes.append(Change(ChangeType.SYNC, name, name))
            push_state_updates.append((name, digest))
            logger.debug(f"  Synced {clickable_path(file_path)}")
        elif not name.startswith("_"):
            try:
                # Store the path before unlinking it so clickable_path can see it exists
                p = file_path
                file_path.unlink()
                removed_names.append(name)
                result.changes.append(Change(ChangeType.DELETE, name, name))
                logger.debug(f"  Deleted orphan {clickable_path(p)}")
            except Exception as e:
                result.errors.append(str(e))

    if push_state_updates:
        db_port.set_media_push_state_bulk(push_state_updates)
    if fingerprint_updates:
        db_port.set_media_fingerprints_bulk(fingerprint_updates)
    if removed_names:
        db_port.remove_media_fingerprints_by_names(removed_names)
        db_port.remove_media_push_state_by_names(removed_names)

    logger.debug(f"Skipped {skipped_pushes} unchanged media pushes")
    return result


def sync_media_from_anki(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    collection_dir: Path,
    db_port: SQLiteDbAdapter,
) -> MediaSyncResult:
    """Pull missing media from Anki."""
    result = MediaSyncResult()
    media_root = collection_dir / LOCAL_MEDIA_DIR
    media_root.mkdir(parents=True, exist_ok=True)

    referenced = _collect_referenced_media(fs_port, db_port, collection_dir)

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

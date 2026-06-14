"""Use Case: Synchronize Media Files with Anki."""

import logging
import re
from pathlib import Path
from urllib.parse import unquote

from blake3 import blake3
from rich.markup import escape as rich_escape

from ankiops.anki import Anki
from ankiops.collection import LOCAL_MEDIA_DIR
from ankiops.console import clickable_path
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
)
from ankiops.sync.report import Change, ChangeType, SyncReport
from ankiops.sync.state import SyncState

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
            # Markdown paths are in group 1 or 2. The other patterns use group 1.
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


def _markdown_cache_key(cache_root: Path, md_file: Path) -> str:
    return str(md_file.relative_to(cache_root))


def extract_media_references(text: str) -> set[str]:
    return _extract_media_references(text)


def calculate_blake3(file_path: Path) -> str:
    hash_state = blake3()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            hash_state.update(chunk)
    return hash_state.hexdigest(length=4)


def update_references(
    directory: Path,
    rename_map: dict[str, str],
    *,
    md_files: list[Path] | None = None,
) -> int:
    if not rename_map:
        return 0

    updated_files = 0
    pattern = re.compile(
        r"(!\[.*?\]\()(?:<(.+?)>|([^()]+(?:\([^()]*\)[^()]*)*))(\)(?:\{[^}]*\})?)"
        r'|(src=["\'])(.+?)(["\'])'
        r"|(\[sound:)(.+?)(\])"
    )

    def replace_callback(match: re.Match) -> str:
        if match.group(1) is not None:
            opener = match.group(1)
            path = match.group(2) or match.group(3)
            suffix = match.group(4)
            is_markdown = True
            media_context = "markdown"
        elif match.group(5) is not None:
            opener, path, suffix = match.group(5), match.group(6), match.group(7)
            is_markdown = False
            media_context = "html"
        else:
            opener, path, suffix = match.group(8), match.group(9), match.group(10)
            is_markdown = False
            media_context = "sound"

        decoded_path = unquote(path)
        lookup_path = decoded_path.strip("<>").replace("\\", "/")
        if lookup_path not in rename_map:
            return match.group(0)

        new_path = rename_map[lookup_path]
        if media_context == "sound" and new_path.startswith(f"{LOCAL_MEDIA_DIR}/"):
            new_path = new_path[len(LOCAL_MEDIA_DIR) + 1 :]
        if is_markdown and not new_path.startswith("<"):
            new_path = f"<{new_path}>"
        return f"{opener}{new_path}{suffix}"

    if md_files is None:
        md_files = DeckSource.local(directory).deck_files()
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        new_content = pattern.sub(replace_callback, content)
        if new_content != content:
            md_file.write_text(new_content, encoding="utf-8")
            updated_files += 1

    return updated_files


def _collect_referenced_media(
    db_port: SyncState,
    collection_dir: Path,
    *,
    cache_root: Path | None = None,
    md_files: list[Path] | None = None,
    prune_cache: bool = True,
) -> set[str]:
    resolved_cache_root = cache_root or collection_dir
    if md_files is None:
        md_files = DeckSource.local(collection_dir).deck_files()
    md_keys = [
        _markdown_cache_key(resolved_cache_root, md_file) for md_file in md_files
    ]
    if prune_cache:
        pruned = db_port.prune_markdown_media_cache(md_keys)
        if pruned:
            logger.debug(f"Pruned media cache for {pruned} stale markdown path(s)")
    cached = db_port.resolve_markdown_media_cache(md_keys)

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
        db_port.upsert_markdown_media_cache(updates)
    if md_files:
        logger.debug(
            f"Media refs cache: {cache_hits} hits, {cache_misses} misses "
            f"across {len(md_files)} markdown files"
        )
    return referenced


def hash_and_update_references(
    db_port: SyncState,
    collection_dir: Path,
    result: SyncReport,
    *,
    cache_root: Path | None = None,
    md_files: list[Path] | None = None,
    prune_cache: bool = True,
) -> tuple[set[str], dict[str, str]]:
    """Hash files, rename them locally, and update markdown."""
    media_root = collection_dir / LOCAL_MEDIA_DIR
    if not media_root.exists():
        return set(), {}

    # 1. Find directly referenced files first
    referenced = _collect_referenced_media(
        db_port,
        collection_dir,
        cache_root=cache_root,
        md_files=md_files,
        prune_cache=prune_cache,
    )

    rename_map: dict[str, str] = {}
    digest_by_name: dict[str, str] = {}
    fingerprint_updates: list[tuple[str, int, int, str, str]] = []
    fingerprint_removals: list[str] = []
    hash_cache_hits = 0
    hash_cache_misses = 0

    media_files = sorted(media_root.glob("*"), key=lambda file_path: file_path.name)
    cache_candidates = [
        file_path.name
        for file_path in media_files
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and (file_path.name in referenced or file_path.name.startswith("_"))
    ]
    cached_fingerprints = db_port.resolve_media_fingerprints(cache_candidates)

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
                digest = calculate_blake3(file_path)
                hash_cache_misses += 1
            new_name = _get_hashed_name(file_path, digest)
            final_path = file_path

            if new_name != file_path.name:
                new_path = file_path.with_name(new_name)
                if new_path.exists():
                    new_digest = calculate_blake3(new_path)
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
                result.add_change(
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

        except Exception as error:
            logger.warning(
                f"Failed to hash media file {clickable_path(file_path)}: "
                f"{rich_escape(str(error))}",
                extra={"markup": True},
            )
            result.errors.append(str(error))

    if fingerprint_updates:
        db_port.upsert_media_fingerprints(fingerprint_updates)
    if fingerprint_removals:
        db_port.delete_media_files(fingerprint_removals)

    if cache_candidates:
        logger.debug(
            f"Media hash cache: {hash_cache_hits} hits, {hash_cache_misses} misses "
            f"across {len(cache_candidates)} candidate files"
        )

    # 3. Update Markdown files in bulk
    if rename_map:
        count = update_references(collection_dir, rename_map, md_files=md_files)
        logger.debug(f"Updated {count} markdown files with {len(rename_map)} renames")
        # 4. Re-collect now-correctly-hashed active references
        final_referenced = _collect_referenced_media(
            db_port,
            collection_dir,
            cache_root=cache_root,
            md_files=md_files,
            prune_cache=False,
        )
        return final_referenced, digest_by_name

    return referenced, digest_by_name


def sync_media_to_anki(
    anki_port: Anki,
    collection_dir: Path,
    db_port: SyncState,
    *,
    cache_root: Path | None = None,
    md_files: list[Path] | None = None,
    prune_cache: bool = True,
) -> SyncReport:
    """Push local media to Anki."""
    result = SyncReport.for_media()
    media_root = (collection_dir / LOCAL_MEDIA_DIR).resolve()

    anki_media_dir = anki_port.get_media_dir().resolve()
    if media_root == anki_media_dir:
        raise ValueError(
            f"Local media directory ({media_root}) is the same as Anki's media "
            "directory. Syncing would rename files inside Anki directly. Aborting."
        )

    active_refs, digest_by_name = hash_and_update_references(
        db_port,
        collection_dir,
        result,
        cache_root=cache_root,
        md_files=md_files,
        prune_cache=prune_cache,
    )
    if not media_root.exists():
        return result

    media_files = sorted(media_root.glob("*"), key=lambda file_path: file_path.name)
    local_media_names = {
        file_path.name
        for file_path in media_files
        if file_path.is_file() and not file_path.name.startswith(".")
    }
    missing_local_refs = sorted(
        name
        for name in active_refs
        if not name.startswith("_") and name not in local_media_names
    )
    if missing_local_refs:
        result.missing += len(missing_local_refs)
        for name in missing_local_refs:
            logger.warning(
                f"Media {name} referenced in markdown but missing in local "
                f"{LOCAL_MEDIA_DIR}/"
            )

    push_candidates = [
        file_path.name
        for file_path in media_files
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and (file_path.name in active_refs or file_path.name.startswith("_"))
    ]
    result.checked = len(active_refs)
    cached_push_state = db_port.resolve_media_push_digests(push_candidates)
    cached_fingerprints = db_port.resolve_media_fingerprints(push_candidates)

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
                    digest = calculate_blake3(file_path)
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
                result.unchanged += 1
                continue

            # Push safely via Port
            anki_port.push_media(file_path, name)
            result.add_change(Change(ChangeType.SYNC, name, name))
            push_state_updates.append((name, digest))
            logger.debug(
                f"  Synced {clickable_path(file_path)}",
                extra={"markup": True},
            )
        elif not name.startswith("_"):
            try:
                # Store the path before unlinking it so clickable_path can see it exists
                removed_file_path = file_path
                file_path.unlink()
                removed_names.append(name)
                result.add_change(Change(ChangeType.DELETE, name, name))
                logger.debug(
                    f"  Deleted orphan {clickable_path(removed_file_path)}",
                    extra={"markup": True},
                )
            except Exception as error:
                result.errors.append(str(error))

    if fingerprint_updates:
        db_port.upsert_media_fingerprints(fingerprint_updates)
    if push_state_updates:
        db_port.upsert_media_push_digests(push_state_updates)
    if removed_names:
        db_port.delete_media_files(removed_names)

    if skipped_pushes:
        noun = "file" if skipped_pushes == 1 else "files"
        logger.debug(
            f"Media push cache: {skipped_pushes} unchanged media {noun} "
            "already present in Anki"
        )
    return result


def sync_media_from_anki(
    anki_port: Anki,
    collection_dir: Path,
    db_port: SyncState,
    *,
    cache_root: Path | None = None,
    md_files: list[Path] | None = None,
    prune_cache: bool = True,
) -> SyncReport:
    """Pull missing media from Anki."""
    result = SyncReport.for_media()
    media_root = collection_dir / LOCAL_MEDIA_DIR
    media_root.mkdir(parents=True, exist_ok=True)

    referenced = _collect_referenced_media(
        db_port,
        collection_dir,
        cache_root=cache_root,
        md_files=md_files,
        prune_cache=prune_cache,
    )
    result.checked = len(referenced)

    for name in referenced:
        target = media_root / name
        if not target.exists():
            success = anki_port.pull_media(name, target)
            if success:
                result.add_change(Change(ChangeType.SYNC, name, name))
                logger.debug(
                    f"  Pulled {clickable_path(target)} from Anki",
                    extra={"markup": True},
                )
            else:
                result.missing += 1
                logger.warning(f"Media {name} missing in Anki")
                result.add_change(Change(ChangeType.SKIP, name, name))
        else:
            result.unchanged += 1
            result.add_change(Change(ChangeType.SKIP, name, name))

    return result


def _combine_media_result(target: SyncReport, source: SyncReport) -> None:
    target.changes.extend(source.changes)
    target.errors.extend(source.errors)
    target.checked += source.checked
    target.unchanged += source.unchanged
    target.missing += source.missing


def _prune_media_cache_for_sources(
    db_port: SyncState,
    collection_dir: Path,
    sources: list[DeckSource],
) -> None:
    md_keys = [
        _markdown_cache_key(collection_dir, md_file)
        for source in sources
        for md_file in source.deck_files()
    ]
    pruned = db_port.prune_markdown_media_cache(md_keys)
    if pruned:
        logger.debug(f"Pruned media cache for {pruned} stale markdown path(s)")


def sync_all_media_to_anki(
    anki_port: Anki,
    collection_dir: Path,
    db_port: SyncState,
) -> SyncReport:
    """Push media from all active sync sources to Anki."""
    sources = discover_deck_sources(collection_dir)
    _prune_media_cache_for_sources(db_port, collection_dir, sources)
    combined = SyncReport.for_media()
    for source in sources:
        source_result = sync_media_to_anki(
            anki_port,
            source.root,
            db_port,
            cache_root=collection_dir,
            md_files=source.deck_files(),
            prune_cache=False,
        )
        _combine_media_result(combined, source_result)
    return combined


def sync_all_media_from_anki(
    anki_port: Anki,
    collection_dir: Path,
    db_port: SyncState,
) -> SyncReport:
    """Pull referenced media for all active sync sources from Anki."""
    sources = discover_deck_sources(collection_dir)
    _prune_media_cache_for_sources(db_port, collection_dir, sources)
    combined = SyncReport.for_media()
    for source in sources:
        source_result = sync_media_from_anki(
            anki_port,
            source.root,
            db_port,
            cache_root=collection_dir,
            md_files=source.deck_files(),
            prune_cache=False,
        )
        _combine_media_result(combined, source_result)
    return combined


def format_media_status(media_result: SyncReport, *, from_anki: bool) -> str:
    checked = media_result.checked
    summary = media_result.summary

    if checked == 0:
        return "Media: no referenced files"

    if media_result.missing:
        if from_anki:
            return (
                f"Media: {checked} files checked — "
                f"{summary.synced} pulled, {media_result.missing} missing in Anki"
            )
        return (
            f"Media: {checked} files checked — "
            f"{summary.format()}, {media_result.missing} missing locally"
        )

    return f"Media: {checked} files checked — {summary.format()}"

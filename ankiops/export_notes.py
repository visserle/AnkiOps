"""Use Case: Export Anki Notes to Markdown."""

import logging
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import (
    NOTE_SEPARATOR,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.fingerprints import note_fingerprint
from ankiops.fs import FileSystemAdapter
from ankiops.models import (
    AnkiNote,
    Change,
    ChangeType,
    CollectionResult,
    MarkdownFile,
    Note,
    NoteTypeConfig,
    SyncResult,
)

logger = logging.getLogger(__name__)


def _from_html(
    anki_note: AnkiNote,
    config: NoteTypeConfig,
    fs_port: FileSystemAdapter,
) -> Note:
    """Convert an AnkiNote into a Domain Note using the FS Port."""
    fields = {}
    for field_config in config.fields:
        if field_config.name == "AnkiOps Key":
            continue
        if field_config.name in anki_note.fields:
            md_val = fs_port.convert_to_markdown(anki_note.fields[field_config.name])
            if md_val:
                fields[field_config.name] = md_val

    note_key = anki_note.fields.get("AnkiOps Key", "").strip()
    return Note(
        note_key=note_key if note_key else None,
        note_type=anki_note.note_type,
        fields=fields,
    )


@dataclass(frozen=True)
class _ResolvedDeckNote:
    note_key: str
    note_id: int
    note: Note
    change: Change | None


def _load_deck_markdown_state(
    deck_name: str,
    existing_file_path: Path | None,
    collection_dir: Path,
    fs_port: FileSystemAdapter,
) -> tuple[Path, MarkdownFile, bool]:
    if existing_file_path and existing_file_path.exists():
        return existing_file_path, fs_port.read_markdown_file(existing_file_path), False

    file_path = collection_dir / f"{deck_name_to_file_stem(deck_name)}.md"
    if file_path.exists():
        return file_path, fs_port.read_markdown_file(file_path), False

    fs_port.write_markdown_file(file_path, "")
    return file_path, fs_port.read_markdown_file(file_path), True


def _resolve_deck_notes(
    anki_notes: list[AnkiNote],
    config_by_name: dict[str, NoteTypeConfig],
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    result: SyncResult,
    note_keys_by_id: dict[int, str],
    pending_note_mappings: list[tuple[str, int]],
    note_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_fingerprints: list[tuple[str, str, str]],
    local_notes_by_note_key: dict[str, Note],
    local_notes_by_content: dict[str, Note],
) -> tuple[list[_ResolvedDeckNote], list[str]]:
    errors: list[str] = []
    resolved_notes: list[_ResolvedDeckNote] = []

    def _queue_note_mapping(note_key: str, note_id: int) -> None:
        if note_keys_by_id.get(note_id) == note_key:
            return
        note_keys_by_id[note_id] = note_key
        pending_note_mappings.append((note_key, note_id))

    def _queue_fingerprint(note_key: str, md_hash: str, anki_hash: str) -> None:
        if note_fingerprints_by_note_key.get(note_key) == (md_hash, anki_hash):
            return
        note_fingerprints_by_note_key[note_key] = (md_hash, anki_hash)
        pending_fingerprints.append((note_key, md_hash, anki_hash))

    for anki_note in anki_notes:
        note_key = note_keys_by_id.get(anki_note.note_id)
        embedded_note_key = anki_note.fields.get("AnkiOps Key", "").strip()
        current_anki_hash = note_fingerprint(anki_note.note_type, anki_note.fields)

        # Stale note_id->note_key mapping: prefer embedded note_key from Anki note.
        if note_key and embedded_note_key and note_key != embedded_note_key:
            _queue_note_mapping(embedded_note_key, anki_note.note_id)
            note_key = embedded_note_key
        elif not note_key and embedded_note_key:
            _queue_note_mapping(embedded_note_key, anki_note.note_id)
            note_key = embedded_note_key

        local_match = local_notes_by_note_key.get(note_key) if note_key else None
        if note_key and local_match:
            local_md_hash = note_fingerprint(local_match.note_type, local_match.fields)
            cached = note_fingerprints_by_note_key.get(note_key)
            if cached == (local_md_hash, current_anki_hash):
                change = Change(
                    ChangeType.SKIP, anki_note.note_id, local_match.identifier
                )
                result.add_change(change)
                resolved_notes.append(
                    _ResolvedDeckNote(
                        note_key=note_key,
                        note_id=anki_note.note_id,
                        note=local_match,
                        change=change,
                    )
                )
                _queue_fingerprint(note_key, local_md_hash, current_anki_hash)
                continue

        cfg = config_by_name.get(anki_note.note_type)
        if not cfg:
            errors.append(
                f"Unknown note type {anki_note.note_type} for note {anki_note.note_id}"
            )
            continue

        domain_note = _from_html(anki_note, cfg, fs_port)

        if not note_key:
            # Match by content when no mapped/embedded note_key exists.
            first_line = domain_note.first_field_line()
            match = local_notes_by_content.get(first_line)
            if match and match.note_key:
                note_key = match.note_key
                _queue_note_mapping(note_key, anki_note.note_id)
            else:
                note_key = db_port.generate_note_key()
                _queue_note_mapping(note_key, anki_note.note_id)

        domain_note.note_key = note_key
        local_match = local_notes_by_note_key.get(note_key)
        if local_match and local_match.fields == domain_note.fields:
            change = Change(ChangeType.SKIP, anki_note.note_id, domain_note.identifier)
            result.add_change(change)
            resolved_notes.append(
                _ResolvedDeckNote(
                    note_key=note_key,
                    note_id=anki_note.note_id,
                    note=local_match,
                    change=change,
                )
            )
            md_hash = note_fingerprint(local_match.note_type, local_match.fields)
            _queue_fingerprint(note_key, md_hash, current_anki_hash)
            continue

        if not local_match:
            change = Change(
                ChangeType.CREATE,
                anki_note.note_id,
                domain_note.identifier,
            )
            result.add_change(change)
            logger.debug(f"  Created {domain_note.identifier}")
        else:
            change = Change(
                ChangeType.UPDATE,
                anki_note.note_id,
                domain_note.identifier,
            )
            result.add_change(change)
            logger.debug(f"  Updated {domain_note.identifier}")

        resolved_notes.append(
            _ResolvedDeckNote(
                note_key=note_key,
                note_id=anki_note.note_id,
                note=domain_note,
                change=change,
            )
        )
        md_hash = note_fingerprint(domain_note.note_type, domain_note.fields)
        _queue_fingerprint(note_key, md_hash, current_anki_hash)

    return resolved_notes, errors


def _order_resolved_notes(
    resolved_notes: list[_ResolvedDeckNote],
    existing_notes: list[Note],
    is_first_export: bool,
) -> list[Note]:
    if is_first_export:
        return [
            resolved.note
            for resolved in sorted(
                resolved_notes,
                key=lambda resolved_note: resolved_note.note_id,
            )
        ]

    resolved_by_note_key = {
        resolved.note_key: resolved for resolved in resolved_notes if resolved.note_key
    }
    consumed_note_keys: set[str] = set()
    ordered_notes: list[Note] = []

    for existing_note in existing_notes:
        if not existing_note.note_key:
            continue
        resolved = resolved_by_note_key.get(existing_note.note_key)
        if not resolved:
            continue
        ordered_notes.append(resolved.note)
        consumed_note_keys.add(resolved.note_key)

    remaining = sorted(
        [
            resolved
            for resolved in resolved_notes
            if resolved.note_key not in consumed_note_keys
        ],
        key=lambda resolved: resolved.note_id,
    )
    ordered_notes.extend(resolved.note for resolved in remaining)
    return ordered_notes


def _render_notes_to_markdown(
    notes: list[Note],
    config_by_name: dict[str, NoteTypeConfig],
) -> str:
    content_parts: list[str] = []

    for note in notes:
        parts = []
        if note.note_key:
            parts.append(f"<!-- note_key: {note.note_key} -->")

        cfg = config_by_name[note.note_type]
        for field_config in cfg.fields:
            if (
                field_config.prefix
                and field_config.name in note.fields
                and note.fields[field_config.name]
            ):
                lines = note.fields[field_config.name].split("\n")
                parts.append(f"{field_config.prefix} {lines[0]}")
                if len(lines) > 1:
                    parts.extend(lines[1:])

        content_parts.append("\n".join(parts))

    return NOTE_SEPARATOR.join(content_parts) + "\n"


def _sync_deck(
    deck_name: str,
    anki_notes: list[AnkiNote],
    config_by_name: dict[str, NoteTypeConfig],
    existing_file_path: Path | None,
    collection_dir: Path,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    note_keys_by_id: dict[int, str],
    pending_note_mappings: list[tuple[str, int]],
    note_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_fingerprints: list[tuple[str, str, str]],
) -> SyncResult:
    result = SyncResult.for_notes(name=deck_name, file_path=existing_file_path)
    file_path, fs, is_first_export = _load_deck_markdown_state(
        deck_name=deck_name,
        existing_file_path=existing_file_path,
        collection_dir=collection_dir,
        fs_port=fs_port,
    )
    result.file_path = file_path

    local_notes_by_note_key = {
        local_note.note_key: local_note
        for local_note in fs.notes
        if local_note.note_key
    }
    local_notes_by_content = {
        local_note.first_field_line(): local_note
        for local_note in fs.notes
        if not local_note.note_key
    }

    resolved_notes, resolve_errors = _resolve_deck_notes(
        anki_notes=anki_notes,
        config_by_name=config_by_name,
        fs_port=fs_port,
        db_port=db_port,
        result=result,
        note_keys_by_id=note_keys_by_id,
        pending_note_mappings=pending_note_mappings,
        note_fingerprints_by_note_key=note_fingerprints_by_note_key,
        pending_fingerprints=pending_fingerprints,
        local_notes_by_note_key=local_notes_by_note_key,
        local_notes_by_content=local_notes_by_content,
    )
    result.errors.extend(resolve_errors)

    final_notes = _order_resolved_notes(
        resolved_notes=resolved_notes,
        existing_notes=fs.notes,
        is_first_export=is_first_export,
    )
    final_text = _render_notes_to_markdown(
        notes=final_notes,
        config_by_name=config_by_name,
    )

    if final_text.strip() != fs.raw_content.strip():
        fs_port.write_markdown_file(result.file_path, final_text)

    # Detect deleted notes
    final_note_keys = {
        final_note.note_key for final_note in final_notes if final_note.note_key
    }
    for old_note in fs.notes:
        if old_note.note_key and old_note.note_key not in final_note_keys:
            # Note was deleted in Anki
            db_port.remove_note_by_note_key(old_note.note_key)
            result.add_change(Change(ChangeType.DELETE, None, old_note.identifier))
            logger.debug(f"  Deleted {old_note.identifier}")

    result.materialize_changes(
        order=(
            ChangeType.DELETE,
            ChangeType.CREATE,
            ChangeType.UPDATE,
            ChangeType.SKIP,
            ChangeType.MOVE,
        )
    )
    return result


def export_collection(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    note_types_dir: Path,
) -> CollectionResult:
    configs = fs_port.load_note_type_configs(note_types_dir)
    config_by_name = {config.name: config for config in configs}
    with db_port.transaction():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()

        all_note_ids = anki_port.fetch_all_note_ids([config.name for config in configs])
        anki_notes = anki_port.fetch_notes_info(all_note_ids)
        note_keys_by_id = db_port.get_note_keys_bulk(all_note_ids)
        pending_note_mappings: list[tuple[str, int]] = []
        note_key_candidates = set(note_keys_by_id.values())
        for anki_note in anki_notes.values():
            embedded_note_key = anki_note.fields.get("AnkiOps Key", "").strip()
            if embedded_note_key:
                note_key_candidates.add(embedded_note_key)
        note_fingerprints_by_note_key = db_port.get_note_fingerprints_bulk(
            note_key_candidates
        )
        pending_fingerprints: list[tuple[str, str, str]] = []

        # Fetch cards info to map notes to decks
        all_card_ids = []
        for anki_note in anki_notes.values():
            all_card_ids.extend(anki_note.card_ids)
        anki_cards = anki_port.fetch_cards_info(all_card_ids)

        notes_by_deck = {}
        for note_id, anki_note in anki_notes.items():
            if not anki_note.card_ids:
                continue
            primary_card_info = anki_cards.get(anki_note.card_ids[0])
            if not primary_card_info:
                continue
            deck_name = primary_card_info.get("deckName")
            notes_by_deck.setdefault(deck_name, []).append(anki_note)

        md_files = fs_port.find_markdown_files(collection_dir)
        file_map_by_name = {}
        for md_file in md_files:
            file_map_by_name[md_file.stem] = md_file

        results = []

        for deck_name, notes in notes_by_deck.items():
            if deck_name == "Default":
                continue

            deck_id = deck_ids_by_name[deck_name]

            # Rename detection: does this deck_id exist under a different name?
            old_name = db_port.get_deck_name(deck_id)
            safe_name = deck_name_to_file_stem(deck_name)
            if old_name and old_name != deck_name:
                old_safe = deck_name_to_file_stem(old_name)
                if old_safe in file_map_by_name:
                    old_path = file_map_by_name[old_safe]
                    new_path = old_path.parent / f"{safe_name}.md"
                    old_path.rename(new_path)
                    file_map_by_name[safe_name] = new_path
                    del file_map_by_name[old_safe]
                    logger.info(f"Deck renamed: '{old_name}' → '{deck_name}'")

            target_file = file_map_by_name.get(safe_name)

            sync_result = _sync_deck(
                deck_name,
                notes,
                config_by_name,
                target_file,
                collection_dir,
                fs_port,
                db_port,
                note_keys_by_id,
                pending_note_mappings,
                note_fingerprints_by_note_key,
                pending_fingerprints,
            )
            db_port.set_deck(deck_name, deck_id)
            results.append(sync_result)
            summary = sync_result.summary
            if summary.format() != "no changes":
                logger.debug(f"Deck '{deck_name}': {summary.format()}")

        if pending_note_mappings:
            db_port.set_notes_bulk(pending_note_mappings)
        if pending_fingerprints:
            db_port.set_note_fingerprints_bulk(pending_fingerprints)
        db_port.prune_orphan_note_fingerprints()

        extra_changes = []
        active_files = {
            sync_result.file_path for sync_result in results if sync_result.file_path
        }
        for md_file in md_files:
            # A prior deck-rename step can move this file already.
            if not md_file.exists():
                continue
            if md_file not in active_files:
                db_port.remove_deck(file_stem_to_deck_name(md_file.stem))
                md_file.unlink()
                extra_changes.append(
                    Change(ChangeType.DELETE, None, f"file: {md_file.name}")
                )

        db_port.save()
        return CollectionResult.for_export(results=results, extra_changes=extra_changes)

"""Use Case: Import Markdown Notes into Anki."""

import logging
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import file_stem_to_deck_name
from ankiops.db import SQLiteDbAdapter
from ankiops.fingerprints import note_fingerprint
from ankiops.fs import FileSystemAdapter
from ankiops.markdown_format import (
    NOTE_SEPARATOR,
    format_note_key_comment,
    format_note_type_comment,
    is_code_fence_line,
    is_note_type_comment,
    parse_note_key_comment,
)
from ankiops.models import (
    ANKIOPS_KEY_FIELD,
    AnkiNote,
    Change,
    ChangeType,
    CollectionResult,
    MarkdownFile,
    Note,
    NoteTypeConfig,
    ProtectedNoteGroup,
    SyncResult,
    UntrackedDeck,
)
from ankiops.note_identity import resolve_import_note_identity
from ankiops.sources import (
    SyncSource,
    discover_sync_sources,
    load_configs_for_sources,
    markdown_files_for_source,
)
from ankiops.tags import parse_tags_comment

logger = logging.getLogger(__name__)


@dataclass
class _PendingWrite:
    file_state: MarkdownFile
    note_key_assignments: list[tuple[Note, str]]


@dataclass(frozen=True)
class _SourceMarkdownFile:
    source: SyncSource
    file_state: MarkdownFile


@dataclass(frozen=True)
class _NoteTypeConversion:
    note_id: int
    note_key: str
    old_model: str
    new_model: str
    entity_repr: str
    md_hash: str
    import_anki_hash: str


def _collab_scope(note_type: str) -> str | None:
    parts = note_type.split("/")
    if len(parts) == 4 and parts[0] == "collab":
        return "/".join(parts[:3])
    return None


def _format_cross_collab_conversion_error(
    *, note_key: str, markdown_note_type: str, anki_note_type: str
) -> str:
    return (
        f"Cannot convert note_key {note_key} from '{anki_note_type}' to "
        f"'{markdown_note_type}': the Anki note already belongs to a different "
        "collab source. Use the matching collab source or resolve the note "
        "ownership manually."
    )


def _find_first_markdown_line_index(note: Note, lines: list[str]) -> int:
    """Find the line index of the first field line within a note block."""
    first_value = note.first_field_line()
    if not first_value:
        return next(
            (line_index for line_index, line in enumerate(lines) if line.strip()), 0
        )

    first_value_stripped = first_value.strip()
    if not first_value_stripped:
        return next(
            (line_index for line_index, line in enumerate(lines) if line.strip()), 0
        )

    # Search the note block for a line containing this value with a field label.
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(first_value_stripped) and ":" in stripped:
            # Verify this is a labeled line (e.g. "Q: Question 1").
            try:
                idx = stripped.rindex(first_value_stripped)
                label_part = stripped[:idx].rstrip()
                if label_part.endswith(":"):
                    return line_index
            except ValueError:
                pass

    return next(
        (line_index for line_index, line in enumerate(lines) if line.strip()), 0
    )


def _find_tags_comment_index(lines: list[str]) -> int | None:
    in_code_block = False
    for line_index, line in enumerate(lines):
        stripped = line.lstrip()
        if is_code_fence_line(stripped):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if parse_tags_comment(line) is not None:
            return line_index
    return None


def _find_managed_metadata_indices(lines: list[str]) -> set[int]:
    indices: set[int] = set()
    in_code_block = False
    for line_index, line in enumerate(lines):
        stripped = line.lstrip()
        if is_code_fence_line(stripped):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if parse_note_key_comment(line) is not None or is_note_type_comment(line):
            indices.add(line_index)
    return indices


def _upsert_note_metadata_in_block(block: str, note: Note, note_key: str | None) -> str:
    """Insert or replace managed metadata in a single markdown note block."""
    lines = block.split("\n")
    metadata_indices = _find_managed_metadata_indices(lines)
    content_lines = [
        line
        for line_index, line in enumerate(lines)
        if line_index not in metadata_indices
    ]

    metadata_lines = []
    if note_key:
        metadata_lines.append(format_note_key_comment(note_key))
    metadata_lines.append(format_note_type_comment(note.note_type))

    tag_idx = _find_tags_comment_index(content_lines)
    insert_idx = (
        tag_idx
        if tag_idx is not None
        else _find_first_markdown_line_index(note, content_lines)
    )
    content_lines[insert_idx:insert_idx] = metadata_lines
    return "\n".join(content_lines)


def _flush_writes(fs_port: FileSystemAdapter, writes: list[_PendingWrite]) -> None:
    for pending_write in writes:
        note_key_by_note = {
            id(note): note_key for note, note_key in pending_write.note_key_assignments
        }
        blocks = pending_write.file_state.raw_content.split(NOTE_SEPARATOR)
        out_blocks: list[str] = []
        note_idx = 0
        changed = False

        for block in blocks:
            stripped_block = block.strip()
            if not stripped_block or not stripped_block.replace("-", ""):
                out_blocks.append(block)
                continue

            if note_idx >= len(pending_write.file_state.notes):
                out_blocks.append(block)
                continue

            note = pending_write.file_state.notes[note_idx]
            note_idx += 1
            note_key = note_key_by_note.get(id(note), note.note_key)

            updated_block = _upsert_note_metadata_in_block(block, note, note_key)
            changed = changed or updated_block != block
            out_blocks.append(updated_block)

        if note_idx != len(pending_write.file_state.notes):
            raise ValueError(
                "Failed to align parsed notes with markdown blocks in "
                f"{pending_write.file_state.file_path.name}"
            )

        if changed:
            fs_port.write_markdown_file(
                pending_write.file_state.file_path, NOTE_SEPARATOR.join(out_blocks)
            )


def _to_html(
    note: Note, config: NoteTypeConfig, converter: FileSystemAdapter
) -> dict[str, str]:
    html = {
        name: converter.convert_to_html(content)
        for name, content in note.fields.items()
    }
    for field_config in config.fields:
        if field_config.label is None:
            continue
        html.setdefault(field_config.name, "")
    return html


def _html_match(html: dict[str, str], anki: AnkiNote) -> bool:
    return all(
        anki.fields.get(field_name) == field_value
        for field_name, field_value in html.items()
    )


def _anki_note_match(
    html: dict[str, str], tags: tuple[str, ...], anki: AnkiNote
) -> bool:
    return _html_match(html, anki) and anki.tags == tags


def _group_note_ids_by_deck_name(anki_cards: dict[int, dict]) -> dict[str, set[int]]:
    note_ids_by_deck_name: dict[str, set[int]] = {}
    for card_info in anki_cards.values():
        deck_name = card_info.get("deckName")
        note_id = card_info.get("note")
        if deck_name and note_id:
            note_ids_by_deck_name.setdefault(deck_name, set()).add(note_id)
    return note_ids_by_deck_name


def _remove_note_ids_from_deck_membership(
    note_ids_by_deck_name: dict[str, set[int]],
    note_ids_to_remove: set[int],
) -> None:
    if not note_ids_to_remove:
        return

    empty_decks: list[str] = []
    for deck_name, note_ids in note_ids_by_deck_name.items():
        note_ids.difference_update(note_ids_to_remove)
        if not note_ids:
            empty_decks.append(deck_name)

    for deck_name in empty_decks:
        note_ids_by_deck_name.pop(deck_name, None)


def _collect_membership_affected_note_ids(results: list[SyncResult]) -> set[int]:
    affected_note_ids: set[int] = set()
    for sync_result in results:
        for change in sync_result.changes:
            if not change.change_type.affects_membership:
                continue
            if isinstance(change.entity_id, int):
                affected_note_ids.add(change.entity_id)
    return affected_note_ids


def _refresh_membership_for_affected_notes(
    *,
    affected_note_ids: set[int],
    anki_port: AnkiAdapter,
    note_ids_by_deck_name: dict[str, set[int]],
) -> None:
    if not affected_note_ids:
        return

    # Clear stale memberships first, then rebuild only for surviving notes.
    _remove_note_ids_from_deck_membership(note_ids_by_deck_name, affected_note_ids)

    refreshed_notes = anki_port.fetch_notes_info(sorted(affected_note_ids))
    refreshed_card_ids: list[int] = []
    for refreshed_note in refreshed_notes.values():
        refreshed_card_ids.extend(refreshed_note.card_ids)

    refreshed_cards = anki_port.fetch_cards_info(refreshed_card_ids)
    refreshed_grouped = _group_note_ids_by_deck_name(refreshed_cards)
    for deck_name, note_ids in refreshed_grouped.items():
        note_ids_by_deck_name.setdefault(deck_name, set()).update(note_ids)


@dataclass(frozen=True)
class _DeckContext:
    deck_name: str
    needs_create_deck: bool


def _resolve_deck_context(
    fs: MarkdownFile,
    deck_ids_by_name: dict[str, int],
    db_port: SQLiteDbAdapter,
) -> _DeckContext:
    deck_name = file_stem_to_deck_name(fs.file_path.stem)
    resolved_id = deck_ids_by_name.get(deck_name)
    if not resolved_id:
        return _DeckContext(deck_name=deck_name, needs_create_deck=True)

    db_port.upsert_deck(deck_name, resolved_id)
    return _DeckContext(deck_name=deck_name, needs_create_deck=False)


def _sync_keyed_note(
    *,
    parsed_note: Note,
    cfg: NoteTypeConfig,
    md_hash: str,
    deck_name: str,
    fs_port: FileSystemAdapter,
    anki_notes: dict[int, AnkiNote],
    anki_cards: dict[int, dict],
    note_ids_by_note_key: dict[str, int],
    note_import_fingerprints_by_note_key: dict[str, tuple[str, str]],
    result: SyncResult,
    cards_to_move: list[int],
    moved_from_decks: set[str],
    note_type_conversions: list[_NoteTypeConversion],
    queue_note_mapping,
    clear_note_mapping,
    queue_import_fingerprint,
) -> None:
    note_key = parsed_note.note_key
    if note_key is None:
        return

    note_id = note_ids_by_note_key.get(note_key)
    anki_note = anki_notes.get(note_id) if note_id else None

    # Keyed import operates on the preloaded managed-note snapshot.
    # If mapped note_id is missing from that snapshot, treat local mapping as stale.
    if note_id and not anki_note:
        clear_note_mapping(note_key)
        note_id = None

    # Keyed sync trusts a non-empty embedded key field as source of truth.
    # If mapped note_id points to a different non-empty key, treat mapping as stale.
    # Empty embedded keys can receive the Markdown note_key during this import.
    if anki_note and note_id:
        embedded_note_key = anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()
        if embedded_note_key and embedded_note_key != note_key:
            clear_note_mapping(note_key)
            note_id = None
            anki_note = None

    if not anki_note:
        html_fields = _to_html(parsed_note, cfg, fs_port)
        html_fields[ANKIOPS_KEY_FIELD.name] = note_key
        result.add_change(
            Change(
                ChangeType.CREATE,
                None,
                parsed_note.identifier,
                {
                    "note": parsed_note,
                    "html_fields": html_fields,
                    "note_key": note_key,
                    "md_hash": md_hash,
                    "import_anki_hash": note_fingerprint(
                        parsed_note.note_type,
                        html_fields,
                        tags=parsed_note.tags,
                    ),
                    "tags": parsed_note.tags,
                },
            )
        )
        logger.debug(
            "  Create Anki note for existing note_key=%s (no mapped note_id found)",
            note_key,
        )
        return

    cards_to_move_for_note = []
    for card_id in anki_note.card_ids:
        card_info = anki_cards.get(card_id)
        if card_info and card_info.get("deckName") != deck_name:
            cards_to_move_for_note.append(card_id)
            source_deck = card_info.get("deckName")
            if source_deck:
                moved_from_decks.add(source_deck)
    if cards_to_move_for_note:
        result.add_change(
            Change(
                ChangeType.MOVE,
                note_id,
                parsed_note.identifier,
                {"cards": cards_to_move_for_note},
            )
        )
        cards_to_move.extend(cards_to_move_for_note)

    needs_note_type_conversion = anki_note.note_type != parsed_note.note_type
    current_anki_hash = note_fingerprint(
        anki_note.note_type,
        anki_note.fields,
        tags=anki_note.tags,
    )
    cached = note_import_fingerprints_by_note_key.get(note_key)
    if not needs_note_type_conversion and cached == (md_hash, current_anki_hash):
        result.add_change(Change(ChangeType.SKIP, note_id, parsed_note.identifier))
        queue_import_fingerprint(note_key, md_hash, current_anki_hash)
        return

    html_fields = _to_html(parsed_note, cfg, fs_port)
    if anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "") != note_key:
        html_fields[ANKIOPS_KEY_FIELD.name] = note_key
    target_anki_hash = note_fingerprint(
        parsed_note.note_type,
        html_fields,
        tags=parsed_note.tags,
    )

    if needs_note_type_conversion:
        anki_scope = _collab_scope(anki_note.note_type)
        markdown_scope = _collab_scope(parsed_note.note_type)
        if anki_scope is not None and anki_scope != markdown_scope:
            result.errors.append(
                _format_cross_collab_conversion_error(
                    note_key=note_key,
                    markdown_note_type=parsed_note.note_type,
                    anki_note_type=anki_note.note_type,
                )
            )
            return
        result.add_change(
            Change(
                ChangeType.CONVERT,
                note_id,
                parsed_note.identifier,
                {
                    "note_key": note_key,
                    "old_model": anki_note.note_type,
                    "new_model": parsed_note.note_type,
                    "md_hash": md_hash,
                    "import_anki_hash": target_anki_hash,
                },
            )
        )
        note_type_conversions.append(
            _NoteTypeConversion(
                note_id=note_id,
                note_key=note_key,
                old_model=anki_note.note_type,
                new_model=parsed_note.note_type,
                entity_repr=parsed_note.identifier,
                md_hash=md_hash,
                import_anki_hash=target_anki_hash,
            )
        )

    if (
        not needs_note_type_conversion
        and _anki_note_match(html_fields, parsed_note.tags, anki_note)
    ):
        result.add_change(Change(ChangeType.SKIP, note_id, parsed_note.identifier))
        queue_import_fingerprint(note_key, md_hash, current_anki_hash)
        return

    if needs_note_type_conversion and _anki_note_match(
        html_fields, parsed_note.tags, anki_note
    ):
        return

    result.add_change(
        Change(
            ChangeType.UPDATE,
            note_id,
            parsed_note.identifier,
            {
                "html_fields": html_fields,
                "note_key": note_key,
                "md_hash": md_hash,
                "import_anki_hash": target_anki_hash,
                "tags": parsed_note.tags,
            },
        )
    )
    logger.debug(
        "  Update Anki note fields for note_key=%s (note_id=%s)",
        note_key,
        note_id,
    )


def _sync_new_note(
    *,
    parsed_note: Note,
    cfg: NoteTypeConfig,
    md_hash: str,
    db_port: SQLiteDbAdapter,
    fs_port: FileSystemAdapter,
    result: SyncResult,
) -> None:
    new_note_key = db_port.generate_note_key()
    html_fields = _to_html(parsed_note, cfg, fs_port)
    html_fields[ANKIOPS_KEY_FIELD.name] = new_note_key
    result.add_change(
        Change(
            ChangeType.CREATE,
            None,
            parsed_note.identifier,
            {
                "note": parsed_note,
                "html_fields": html_fields,
                "note_key": new_note_key,
                "md_hash": md_hash,
                "import_anki_hash": note_fingerprint(
                    parsed_note.note_type,
                    html_fields,
                    tags=parsed_note.tags,
                ),
                "tags": parsed_note.tags,
            },
        )
    )
    logger.debug(
        "  Create Anki note and assign new note_key=%s from unkeyed markdown note",
        new_note_key,
    )


def _collect_orphan_deletes(
    *,
    fs_notes: list[Note],
    note_ids_by_note_key: dict[str, int],
    global_mapped_note_ids: set[int],
    all_anki_note_ids: set[int],
    anki_notes: dict[int, AnkiNote],
    db_port: SQLiteDbAdapter,
    result: SyncResult,
) -> None:
    md_anki_ids = set()
    for parsed_note in fs_notes:
        if parsed_note.note_key:
            note_id = note_ids_by_note_key.get(parsed_note.note_key)
            if note_id:
                md_anki_ids.add(note_id)

    # Orphans
    orphaned = all_anki_note_ids - md_anki_ids
    if global_mapped_note_ids:
        orphaned -= global_mapped_note_ids

    orphan_note_keys = db_port.resolve_note_keys(orphaned)
    for note_id in sorted(orphaned):
        if note_id:
            orphan_note = anki_notes.get(note_id)
            embedded_note_key = ""
            if orphan_note is not None:
                embedded_note_key = orphan_note.fields.get(
                    ANKIOPS_KEY_FIELD.name, ""
                ).strip()

            # Protect unmanaged keyless notes from deletion on import.
            if not embedded_note_key:
                result.protected_keyless_notes += 1
                result.add_change(
                    Change(
                        ChangeType.SKIP,
                        note_id,
                        f"note_id: {note_id}",
                        {"protected_keyless": True},
                    )
                )
                logger.debug(
                    "  Skip delete for unmanaged note_id=%s (missing AnkiOps Key)",
                    note_id,
                )
                continue

            note_key = orphan_note_keys.get(note_id)
            delete_repr = f"note_key: {note_key}" if note_key else f"note_id: {note_id}"
            result.add_change(
                Change(
                    ChangeType.DELETE,
                    note_id,
                    delete_repr,
                    {"note_key": note_key} if note_key else {},
                )
            )
            if note_key:
                logger.debug(
                    "  Delete orphan managed Anki note note_id=%s (note_key=%s)",
                    note_id,
                    note_key,
                )
            else:
                logger.debug(
                    "  Delete orphan managed Anki note note_id=%s (note_key unknown)",
                    note_id,
                )


def _apply_note_type_conversions(
    *,
    anki_port: AnkiAdapter,
    conversions: list[_NoteTypeConversion],
    queue_import_fingerprint,
) -> list[str]:
    if not conversions:
        return []

    by_model_pair: dict[tuple[str, str], list[_NoteTypeConversion]] = {}
    for conversion in conversions:
        by_model_pair.setdefault(
            (conversion.old_model, conversion.new_model),
            [],
        ).append(conversion)

    try:
        for (old_model, new_model), grouped in sorted(by_model_pair.items()):
            anki_port.change_notes_notetype(
                sorted(conversion.note_id for conversion in grouped),
                old_model,
                new_model,
            )
    except Exception as error:
        affected = ", ".join(conversion.entity_repr for conversion in conversions)
        return [f"Failed note type conversion ({affected}): {error}"]

    for conversion in conversions:
        queue_import_fingerprint(
            conversion.note_key,
            conversion.md_hash,
            conversion.import_anki_hash,
        )
    return []


def _apply_changes_and_update_state(
    *,
    deck_context: _DeckContext,
    anki_port: AnkiAdapter,
    db_port: SQLiteDbAdapter,
    result: SyncResult,
    cards_to_move: list[int],
    note_type_conversions: list[_NoteTypeConversion],
    note_ids_by_note_key: dict[str, int],
    global_note_keys: set[str],
    global_mapped_note_ids: set[int],
    note_import_fingerprints_by_note_key: dict[str, tuple[str, str]],
    queue_note_mapping,
    queue_import_fingerprint,
) -> list[tuple[Note, str]]:
    if not deck_context.deck_name:
        return []

    if result.errors:
        return []

    conversion_errors = _apply_note_type_conversions(
        anki_port=anki_port,
        conversions=note_type_conversions,
        queue_import_fingerprint=queue_import_fingerprint,
    )
    if conversion_errors:
        result.errors.extend(conversion_errors)
        return []

    creates = result.changes_for(ChangeType.CREATE)
    updates = result.changes_for(ChangeType.UPDATE)
    deletes = result.changes_for(ChangeType.DELETE)
    created_ids, errors = anki_port.apply_note_changes(
        deck_context.deck_name,
        deck_context.needs_create_deck,
        creates,
        updates,
        deletes,
        cards_to_move,
    )
    result.errors.extend(errors)
    delete_failed = any(err.startswith("Failed delete") for err in errors)

    note_key_assignments: list[tuple[Note, str]] = []

    # Link mapped created IDs
    for note_id, create_change in zip(created_ids, creates):
        if note_id is None:
            continue
        note_key = create_change.context["note_key"]
        queue_note_mapping(note_key, note_id)
        queue_import_fingerprint(
            note_key,
            create_change.context["md_hash"],
            create_change.context["import_anki_hash"],
        )
        create_change.entity_id = note_id
        note_key_assignments.append((create_change.context["note"], note_key))

    for update_change in updates:
        queue_import_fingerprint(
            update_change.context["note_key"],
            update_change.context["md_hash"],
            update_change.context["import_anki_hash"],
        )

    if deletes and not delete_failed:
        delete_note_keys = [
            delete_change.context.get("note_key")
            for delete_change in deletes
            if delete_change.context
        ]
        delete_note_keys = [note_key for note_key in delete_note_keys if note_key]
        if delete_note_keys:
            # note_state row deletion also clears export-direction fingerprints.
            db_port.delete_note_links_by_keys(delete_note_keys)
            for note_key in delete_note_keys:
                removed_note_id = note_ids_by_note_key.pop(note_key, None)
                if note_key in global_note_keys and removed_note_id is not None:
                    global_mapped_note_ids.discard(removed_note_id)
                note_import_fingerprints_by_note_key.pop(note_key, None)

    return note_key_assignments


def _recover_created_deck_mapping(
    *,
    deck_context: _DeckContext,
    anki_port: AnkiAdapter,
    deck_ids_by_name: dict[str, int],
    deck_names_by_id: dict[int, str],
    db_port: SQLiteDbAdapter,
) -> None:
    if not deck_context.needs_create_deck or not deck_context.deck_name:
        return

    new_deck_ids = anki_port.fetch_deck_names_and_ids()
    if deck_context.deck_name in new_deck_ids:
        new_id = new_deck_ids[deck_context.deck_name]
        deck_ids_by_name[deck_context.deck_name] = new_id
        deck_names_by_id[new_id] = deck_context.deck_name
        db_port.upsert_deck(deck_context.deck_name, new_id)


def _sync_file(
    fs: MarkdownFile,
    config_by_name: dict[str, NoteTypeConfig],
    anki_port: AnkiAdapter,
    db_port: SQLiteDbAdapter,
    fs_port: FileSystemAdapter,
    deck_names_by_id: dict[int, str],
    deck_ids_by_name: dict[str, int],
    anki_notes: dict[int, AnkiNote],
    anki_cards: dict[int, dict],
    note_ids_by_note_key: dict[str, int],
    pending_note_mappings: list[tuple[str, int]],
    note_import_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_import_fingerprints: list[tuple[str, str, str]],
    global_note_keys: set[str],
    global_mapped_note_ids: set[int],
    all_anki_note_ids: set[int],
) -> tuple[SyncResult, _PendingWrite, set[str]]:
    deck_context = _resolve_deck_context(fs, deck_ids_by_name, db_port)
    result = SyncResult.for_notes(name=deck_context.deck_name, file_path=fs.file_path)
    cards_to_move: list[int] = []
    moved_from_decks: set[str] = set()
    note_type_conversions: list[_NoteTypeConversion] = []

    def _queue_note_mapping(note_key: str, note_id: int) -> None:
        existing_note_id = note_ids_by_note_key.get(note_key)
        if existing_note_id == note_id:
            return
        note_ids_by_note_key[note_key] = note_id
        if note_key in global_note_keys:
            if existing_note_id is not None:
                global_mapped_note_ids.discard(existing_note_id)
            global_mapped_note_ids.add(note_id)
        pending_note_mappings.append((note_key, note_id))

    def _clear_note_mapping(note_key: str) -> None:
        existing_note_id = note_ids_by_note_key.pop(note_key, None)
        if existing_note_id is None:
            return
        if note_key in global_note_keys:
            global_mapped_note_ids.discard(existing_note_id)
        note_import_fingerprints_by_note_key.pop(note_key, None)
        # note_state row deletion also clears export-direction fingerprints.
        db_port.delete_note_links_by_keys([note_key])

    def _queue_import_fingerprint(note_key: str, md_hash: str, anki_hash: str) -> None:
        if note_import_fingerprints_by_note_key.get(note_key) == (md_hash, anki_hash):
            return
        note_import_fingerprints_by_note_key[note_key] = (md_hash, anki_hash)
        pending_import_fingerprints.append((note_key, md_hash, anki_hash))

    for parsed_note in fs.notes:
        cfg = config_by_name[parsed_note.note_type]
        md_hash = note_fingerprint(
            parsed_note.note_type,
            parsed_note.fields,
            tags=parsed_note.tags,
        )

        if parsed_note.note_key:
            _sync_keyed_note(
                parsed_note=parsed_note,
                cfg=cfg,
                md_hash=md_hash,
                deck_name=deck_context.deck_name,
                fs_port=fs_port,
                anki_notes=anki_notes,
                anki_cards=anki_cards,
                note_ids_by_note_key=note_ids_by_note_key,
                note_import_fingerprints_by_note_key=note_import_fingerprints_by_note_key,
                result=result,
                cards_to_move=cards_to_move,
                moved_from_decks=moved_from_decks,
                note_type_conversions=note_type_conversions,
                queue_note_mapping=_queue_note_mapping,
                clear_note_mapping=_clear_note_mapping,
                queue_import_fingerprint=_queue_import_fingerprint,
            )
        else:
            _sync_new_note(
                parsed_note=parsed_note,
                cfg=cfg,
                md_hash=md_hash,
                db_port=db_port,
                fs_port=fs_port,
                result=result,
            )

    _collect_orphan_deletes(
        fs_notes=fs.notes,
        note_ids_by_note_key=note_ids_by_note_key,
        global_mapped_note_ids=global_mapped_note_ids,
        all_anki_note_ids=all_anki_note_ids,
        anki_notes=anki_notes,
        db_port=db_port,
        result=result,
    )

    note_key_assignments = _apply_changes_and_update_state(
        deck_context=deck_context,
        anki_port=anki_port,
        db_port=db_port,
        result=result,
        cards_to_move=cards_to_move,
        note_type_conversions=note_type_conversions,
        note_ids_by_note_key=note_ids_by_note_key,
        global_note_keys=global_note_keys,
        global_mapped_note_ids=global_mapped_note_ids,
        note_import_fingerprints_by_note_key=note_import_fingerprints_by_note_key,
        queue_note_mapping=_queue_note_mapping,
        queue_import_fingerprint=_queue_import_fingerprint,
    )

    _recover_created_deck_mapping(
        deck_context=deck_context,
        anki_port=anki_port,
        deck_ids_by_name=deck_ids_by_name,
        deck_names_by_id=deck_names_by_id,
        db_port=db_port,
    )

    result.order_changes()
    pending = _PendingWrite(fs, note_key_assignments)
    return result, pending, moved_from_decks


def import_collection(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    note_types_dir: Path,
) -> CollectionResult:
    sources = discover_sync_sources(collection_dir, note_types_dir=note_types_dir)
    source_configs = load_configs_for_sources(sources)
    configs = [
        config
        for source_config in source_configs
        for config in source_config.configs
    ]
    required_note_types = [config.name for config in configs]
    config_by_name = {config.name: config for config in configs}
    fs_port.set_configs(configs)
    source_docs: list[_SourceMarkdownFile] = []
    for source_config in source_configs:
        source_fs = FileSystemAdapter()
        source_fs.set_configs(source_config.configs)
        for md_file in markdown_files_for_source(source_config.source):
            source_docs.append(
                _SourceMarkdownFile(
                    source=source_config.source,
                    file_state=source_fs.read_markdown_file(
                        md_file,
                        context_root=source_config.source.root,
                    ),
                )
            )
    fs_docs = [source_doc.file_state for source_doc in source_docs]

    deck_sources: dict[str, str] = {}
    deck_duplicates: list[str] = []
    for source_doc in source_docs:
        deck_name = file_stem_to_deck_name(source_doc.file_state.file_path.stem)
        previous = deck_sources.get(deck_name)
        current = (
            f"{source_doc.source.display_name}:"
            f"{source_doc.file_state.file_path.name}"
        )
        if previous is not None:
            deck_duplicates.append(
                f"Deck '{deck_name}' is defined by both {previous} and {current}"
            )
            continue
        deck_sources[deck_name] = current
    if deck_duplicates:
        raise ValueError(
            "Aborting import: deck ownership conflicts found: "
            + "; ".join(deck_duplicates)
        )

    global_note_keys = set()
    note_key_sources = {}
    duplicates = []

    for markdown_file in fs_docs:
        for note in markdown_file.notes:
            if note.note_key:
                if note.note_key in note_key_sources:
                    duplicates.append(f"Duplicate note_key {note.note_key}")
                else:
                    note_key_sources[note.note_key] = str(
                        markdown_file.file_path.relative_to(collection_dir)
                    )
                global_note_keys.add(note.note_key)

    if duplicates:
        raise ValueError(f"Aborting import: Duplicates found: {duplicates}")
    note_import_fingerprints_by_note_key = db_port.resolve_import_hashes(
        global_note_keys
    )
    pending_import_fingerprints: list[tuple[str, str, str]] = []

    with db_port.write_tx():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
        initial_deck_names = set(deck_ids_by_name)
        deck_names_by_id = {
            deck_id: deck_name for deck_name, deck_id in deck_ids_by_name.items()
        }

        identity = resolve_import_note_identity(
            anki_port=anki_port,
            db_port=db_port,
            note_keys=global_note_keys,
            required_note_types=required_note_types,
        )
        anki_notes = identity.anki_notes
        note_ids_by_note_key = identity.note_ids_by_note_key
        pending_note_mappings = list(identity.pending_note_mappings)

        global_mapped_note_ids = {
            note_id
            for note_key, note_id in note_ids_by_note_key.items()
            if note_key in global_note_keys and note_id
        }

        # We need to compute anki_cards and group note_ids by deck.
        # A single bulk cardsInfo fetch is faster than many smaller calls.
        all_card_ids = []
        for anki_note in anki_notes.values():
            all_card_ids.extend(anki_note.card_ids)
        anki_cards = anki_port.fetch_cards_info(all_card_ids)

        note_ids_by_deck_name = _group_note_ids_by_deck_name(anki_cards)

        results = []
        pending = []
        md_deck_ids = set()
        markdown_deck_names = {
            file_stem_to_deck_name(markdown_file.file_path.stem)
            for markdown_file in fs_docs
        }
        rename_candidates: set[tuple[str, str]] = set()

        for source_doc in source_docs:
            markdown_file = source_doc.file_state
            # Determine actual anki_deck_note_ids for this specific file
            # We find what the deck name maps to...
            deck_name = file_stem_to_deck_name(markdown_file.file_path.stem)

            file_anki_note_ids = note_ids_by_deck_name.get(deck_name, set())

            sync_result, pending_write, moved_from_decks = _sync_file(
                markdown_file,
                config_by_name,
                anki_port,
                db_port,
                fs_port,
                deck_names_by_id,
                deck_ids_by_name,
                anki_notes,
                anki_cards,
                note_ids_by_note_key,
                pending_note_mappings,
                note_import_fingerprints_by_note_key,
                pending_import_fingerprints,
                global_note_keys,
                global_mapped_note_ids,
                file_anki_note_ids,
            )

            # Track filename-based deck rename candidates, but defer applying
            # mapping cleanup until deck membership is refreshed.
            if (
                sync_result.name
                and sync_result.name not in initial_deck_names
                and len(moved_from_decks) == 1
            ):
                source_deck = next(iter(moved_from_decks))
                if (
                    source_deck != sync_result.name
                    and source_deck not in markdown_deck_names
                ):
                    rename_candidates.add((source_deck, sync_result.name))

            if sync_result.name and sync_result.name in deck_ids_by_name:
                md_deck_ids.add(deck_ids_by_name[sync_result.name])

            results.append(sync_result)
            pending.append(pending_write)
            summary = sync_result.summary
            if summary.format() != "no changes":
                logger.debug(
                    f"File '{markdown_file.file_path.name}': {summary.format()}"
                )

        if pending_note_mappings:
            db_port.upsert_note_links(pending_note_mappings)
        if pending_import_fingerprints:
            db_port.upsert_import_hashes(pending_import_fingerprints)

        _flush_writes(fs_port, pending)

        # Keep membership in sync with applied structural changes without a
        # full collection refresh.
        affected_note_ids = _collect_membership_affected_note_ids(results)
        _refresh_membership_for_affected_notes(
            affected_note_ids=affected_note_ids,
            anki_port=anki_port,
            note_ids_by_deck_name=note_ids_by_deck_name,
        )

        # Finalize deferred rename candidates only when the source deck is
        # fully drained after moves/deletes.
        for source_deck, target_deck in sorted(rename_candidates):
            if source_deck in note_ids_by_deck_name:
                continue
            db_port.delete_deck(source_deck)
            logger.info(
                f"Deck renamed from markdown file: '{source_deck}' -> '{target_deck}'"
            )

        untracked = []
        for deck_name, note_ids in note_ids_by_deck_name.items():
            deck_id = deck_ids_by_name.get(deck_name)
            if deck_id is None or deck_id in md_deck_ids:
                continue
            untracked.append(UntrackedDeck(deck_name, deck_id, list(note_ids)))

    protected_note_groups = [
        ProtectedNoteGroup(sync_result.name or "", sync_result.protected_keyless_notes)
        for sync_result in results
        if sync_result.protected_keyless_notes and sync_result.name
    ]

    return CollectionResult.for_import(
        results=results,
        untracked_decks=untracked,
        protected_note_groups=protected_note_groups,
    )

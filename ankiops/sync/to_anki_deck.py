"""Per-deck Markdown-to-Anki sync engine."""

import logging
from dataclasses import dataclass

from ankiops.anki import Anki
from ankiops.collection import file_stem_to_deck_name
from ankiops.markdown import (
    NOTE_SEPARATOR,
    DeckFile,
    format_note_key_comment,
    format_note_type_comment,
    is_code_fence_line,
    is_note_type_comment,
    parse_note_key_comment,
    parse_tags_comment,
    write_deck_file,
)
from ankiops.markdown_to_html import MarkdownToHTML
from ankiops.note_types import ANKIOPS_KEY_FIELD, NoteType
from ankiops.notes import (
    AnkiNote,
    Note,
    note_fingerprint,
)
from ankiops.sync.report import (
    Change,
    ChangeType,
    SyncReport,
)
from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)


@dataclass
class PendingDeckWrite:
    file_state: DeckFile
    note_key_assignments: list[tuple[Note, str]]


@dataclass(frozen=True)
class _NoteTypeConversion:
    note_id: int
    note_key: str
    old_model: str
    new_model: str
    entity_repr: str
    md_hash: str
    import_anki_hash: str


def _shared_scope(note_type: str) -> str | None:
    parts = note_type.split("/")
    if len(parts) == 4 and parts[0] == "shared":
        return "/".join(parts[:3])
    return None


def _format_cross_shared_conversion_error(
    *, note_key: str, markdown_note_type: str, anki_note_type: str
) -> str:
    return (
        f"Cannot convert note_key {note_key} from '{anki_note_type}' to "
        f"'{markdown_note_type}': the Anki note already belongs to a different "
        "shared source. Use the matching shared source or resolve the note "
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


def flush_deck_metadata_writes(writes: list[PendingDeckWrite]) -> None:
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
            write_deck_file(
                pending_write.file_state.file_path,
                NOTE_SEPARATOR.join(out_blocks),
            )


def _to_html(note: Note, config: NoteType, converter: MarkdownToHTML) -> dict[str, str]:
    html = {name: converter.convert(content) for name, content in note.fields.items()}
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


def group_note_ids_by_deck_name(anki_cards: dict[int, dict]) -> dict[str, set[int]]:
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


def collect_membership_affected_note_ids(results: list[SyncReport]) -> set[int]:
    affected_note_ids: set[int] = set()
    for sync_result in results:
        for change in sync_result.changes:
            if not change.change_type.affects_membership:
                continue
            if isinstance(change.entity_id, int):
                affected_note_ids.add(change.entity_id)
    return affected_note_ids


def refresh_membership_for_affected_notes(
    *,
    affected_note_ids: set[int],
    anki_port: Anki,
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
    refreshed_grouped = group_note_ids_by_deck_name(refreshed_cards)
    for deck_name, note_ids in refreshed_grouped.items():
        note_ids_by_deck_name.setdefault(deck_name, set()).update(note_ids)


@dataclass(frozen=True)
class _DeckContext:
    deck_name: str
    needs_create_deck: bool


def _resolve_deck_context(
    fs: DeckFile,
    deck_ids_by_name: dict[str, int],
    db_port: SyncState,
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
    cfg: NoteType,
    md_hash: str,
    deck_name: str,
    markdown_to_html: MarkdownToHTML,
    anki_notes: dict[int, AnkiNote],
    anki_cards: dict[int, dict],
    note_ids_by_note_key: dict[str, int],
    note_import_fingerprints_by_note_key: dict[str, tuple[str, str]],
    result: SyncReport,
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
        html_fields = _to_html(parsed_note, cfg, markdown_to_html)
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

    html_fields = _to_html(parsed_note, cfg, markdown_to_html)
    if anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "") != note_key:
        html_fields[ANKIOPS_KEY_FIELD.name] = note_key
    target_anki_hash = note_fingerprint(
        parsed_note.note_type,
        html_fields,
        tags=parsed_note.tags,
    )

    if needs_note_type_conversion:
        anki_scope = _shared_scope(anki_note.note_type)
        markdown_scope = _shared_scope(parsed_note.note_type)
        if anki_scope is not None and anki_scope != markdown_scope:
            result.errors.append(
                _format_cross_shared_conversion_error(
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

    if not needs_note_type_conversion and _anki_note_match(
        html_fields, parsed_note.tags, anki_note
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
    cfg: NoteType,
    md_hash: str,
    db_port: SyncState,
    markdown_to_html: MarkdownToHTML,
    result: SyncReport,
) -> None:
    new_note_key = db_port.generate_note_key()
    html_fields = _to_html(parsed_note, cfg, markdown_to_html)
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
    db_port: SyncState,
    result: SyncReport,
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
    anki_port: Anki,
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
    anki_port: Anki,
    db_port: SyncState,
    result: SyncReport,
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
    anki_port: Anki,
    deck_ids_by_name: dict[str, int],
    deck_names_by_id: dict[int, str],
    db_port: SyncState,
) -> None:
    if not deck_context.needs_create_deck or not deck_context.deck_name:
        return

    new_deck_ids = anki_port.fetch_deck_names_and_ids()
    if deck_context.deck_name in new_deck_ids:
        new_id = new_deck_ids[deck_context.deck_name]
        deck_ids_by_name[deck_context.deck_name] = new_id
        deck_names_by_id[new_id] = deck_context.deck_name
        db_port.upsert_deck(deck_context.deck_name, new_id)


def sync_deck_to_anki(
    fs: DeckFile,
    config_by_name: dict[str, NoteType],
    anki_port: Anki,
    db_port: SyncState,
    markdown_to_html: MarkdownToHTML,
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
) -> tuple[SyncReport, PendingDeckWrite, set[str]]:
    deck_context = _resolve_deck_context(fs, deck_ids_by_name, db_port)
    result = SyncReport.for_notes(name=deck_context.deck_name, file_path=fs.file_path)
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
                markdown_to_html=markdown_to_html,
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
                markdown_to_html=markdown_to_html,
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
    pending = PendingDeckWrite(fs, note_key_assignments)
    return result, pending, moved_from_decks

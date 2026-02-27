"""Use Case: Import Markdown Notes into Anki."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import NOTE_SEPARATOR, file_stem_to_deck_name
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
    UntrackedDeck,
)

logger = logging.getLogger(__name__)
_NOTE_KEY_LINE_RE = re.compile(r"^\s*<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->\s*$")


@dataclass
class _PendingWrite:
    file_state: MarkdownFile
    note_key_assignments: list[tuple[Note, str]]


def _format_note_type_mismatch_error(
    *, note_key: str, markdown_note_type: str, anki_note_type: str
) -> str:
    return (
        f"Note type mismatch for note_key: {note_key}: markdown uses "
        f"'{markdown_note_type}' but Anki has '{anki_note_type}'. "
        "Anki cannot convert existing notes between note types. "
        f"Remove this note's key comment (<!-- note_key: {note_key} -->) "
        "to force creating a new note with the new type on the next import."
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

    # Search the note block for a line containing this value with a prefix.
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(first_value_stripped) and ":" in stripped:
            # Verify this is a prefixed line (e.g. "Q: Question 1").
            try:
                idx = stripped.rindex(first_value_stripped)
                prefix_part = stripped[:idx].rstrip()
                if prefix_part.endswith(":"):
                    return line_index
            except ValueError:
                pass

    return next(
        (line_index for line_index, line in enumerate(lines) if line.strip()), 0
    )


def _upsert_note_key_in_block(block: str, note: Note, note_key: str) -> str:
    """Insert or replace note_key in a single markdown note block."""
    note_key_line = f"<!-- note_key: {note_key} -->"
    lines = block.split("\n")

    # Replace an existing note_key comment in place.
    for line_index, line in enumerate(lines):
        if _NOTE_KEY_LINE_RE.match(line):
            if line.strip() == note_key_line:
                return block
            lines[line_index] = note_key_line
            return "\n".join(lines)

    insert_idx = _find_first_markdown_line_index(note, lines)
    lines.insert(insert_idx, note_key_line)
    return "\n".join(lines)


def _flush_writes(fs_port: FileSystemAdapter, writes: list[_PendingWrite]) -> None:
    for pending_write in writes:
        if not pending_write.note_key_assignments:
            continue

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
            note_key = note_key_by_note.get(id(note))
            if note_key is None:
                out_blocks.append(block)
                continue

            updated_block = _upsert_note_key_in_block(block, note, note_key)
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
        if field_config.prefix is None:
            continue
        html.setdefault(field_config.name, "")
    return html


def _html_match(html: dict[str, str], anki: AnkiNote) -> bool:
    return all(
        anki.fields.get(field_name) == field_value
        for field_name, field_value in html.items()
    )


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

    db_port.set_deck(deck_name, resolved_id)
    return _DeckContext(deck_name=deck_name, needs_create_deck=False)


def _sync_keyed_note(
    *,
    parsed_note: Note,
    cfg: NoteTypeConfig,
    md_hash: str,
    deck_name: str,
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    anki_notes: dict[int, AnkiNote],
    anki_cards: dict[int, dict],
    note_ids_by_note_key: dict[str, int],
    note_fingerprints_by_note_key: dict[str, tuple[str, str]],
    result: SyncResult,
    cards_to_move: list[int],
    moved_from_decks: set[str],
    queue_note_mapping,
    queue_fingerprint,
) -> None:
    note_key = parsed_note.note_key
    if note_key is None:
        return

    note_id = note_ids_by_note_key.get(note_key)
    if note_id is None:
        # Recovery
        found = anki_port.find_notes_by_ankiops_note_key(note_key)
        if found:
            note_id = found[0]
            queue_note_mapping(note_key, note_id)

    anki_note = anki_notes.get(note_id) if note_id else None

    # If recovery worked but wasn't in original batch fetch
    if note_id and not anki_note:
        info = anki_port.fetch_notes_info([note_id])
        if info:
            anki_note = info[note_id]
            anki_notes[note_id] = anki_note
        else:
            # Stale note_key->note_id mapping: recover via embedded AnkiOps Key.
            found = anki_port.find_notes_by_ankiops_note_key(note_key)
            if found:
                note_id = found[0]
                queue_note_mapping(note_key, note_id)
                info = anki_port.fetch_notes_info([note_id])
                if info:
                    anki_note = info[note_id]
                    anki_notes[note_id] = anki_note

    if not anki_note:
        html_fields = _to_html(parsed_note, cfg, fs_port)
        html_fields["AnkiOps Key"] = note_key
        result.change_buckets.creates.append(
            Change(
                ChangeType.CREATE,
                None,
                parsed_note.identifier,
                {
                    "note": parsed_note,
                    "html_fields": html_fields,
                    "note_key": note_key,
                    "md_hash": md_hash,
                    "anki_hash": note_fingerprint(parsed_note.note_type, html_fields),
                },
            )
        )
        logger.debug(f"  Create {parsed_note.identifier}")
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
        result.change_buckets.moves.append(
            Change(
                ChangeType.MOVE,
                note_id,
                parsed_note.identifier,
                {"cards": cards_to_move_for_note},
            )
        )
        cards_to_move.extend(cards_to_move_for_note)

    if anki_note.note_type != parsed_note.note_type:
        result.errors.append(
            _format_note_type_mismatch_error(
                note_key=note_key,
                markdown_note_type=parsed_note.note_type,
                anki_note_type=anki_note.note_type,
            )
        )
        return

    current_anki_hash = note_fingerprint(anki_note.note_type, anki_note.fields)
    cached = note_fingerprints_by_note_key.get(note_key)
    if cached == (md_hash, current_anki_hash):
        result.change_buckets.skips.append(
            Change(ChangeType.SKIP, note_id, parsed_note.identifier)
        )
        queue_fingerprint(note_key, md_hash, current_anki_hash)
        return

    html_fields = _to_html(parsed_note, cfg, fs_port)
    if anki_note.fields.get("AnkiOps Key", "") != note_key:
        html_fields["AnkiOps Key"] = note_key

    if _html_match(html_fields, anki_note):
        result.change_buckets.skips.append(
            Change(ChangeType.SKIP, note_id, parsed_note.identifier)
        )
        queue_fingerprint(note_key, md_hash, current_anki_hash)
        return

    target_anki_hash = note_fingerprint(parsed_note.note_type, html_fields)
    result.change_buckets.updates.append(
        Change(
            ChangeType.UPDATE,
            note_id,
            parsed_note.identifier,
            {
                "html_fields": html_fields,
                "note_key": note_key,
                "md_hash": md_hash,
                "anki_hash": target_anki_hash,
            },
        )
    )
    logger.debug(f"  Update {parsed_note.identifier}")


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
    html_fields["AnkiOps Key"] = new_note_key
    result.change_buckets.creates.append(
        Change(
            ChangeType.CREATE,
            None,
            parsed_note.identifier,
            {
                "note": parsed_note,
                "html_fields": html_fields,
                "note_key": new_note_key,
                "md_hash": md_hash,
                "anki_hash": note_fingerprint(parsed_note.note_type, html_fields),
            },
        )
    )
    logger.debug(f"  Create (new) {parsed_note.identifier}")


def _collect_orphan_deletes(
    *,
    fs_notes: list[Note],
    note_ids_by_note_key: dict[str, int],
    global_mapped_note_ids: set[int],
    all_anki_note_ids: set[int],
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

    orphan_note_keys = db_port.get_note_keys_bulk(orphaned)
    for note_id in sorted(orphaned):
        if note_id:
            note_key = orphan_note_keys.get(note_id)
            delete_repr = f"note_key: {note_key}" if note_key else f"note_id: {note_id}"
            result.change_buckets.deletes.append(
                Change(
                    ChangeType.DELETE,
                    note_id,
                    delete_repr,
                    {"note_key": note_key} if note_key else {},
                )
            )
            logger.debug(f"  Delete {delete_repr}")


def _apply_changes_and_update_state(
    *,
    deck_context: _DeckContext,
    anki_port: AnkiAdapter,
    db_port: SQLiteDbAdapter,
    result: SyncResult,
    cards_to_move: list[int],
    note_ids_by_note_key: dict[str, int],
    global_note_keys: set[str],
    global_mapped_note_ids: set[int],
    note_fingerprints_by_note_key: dict[str, tuple[str, str]],
    queue_note_mapping,
    queue_fingerprint,
) -> list[tuple[Note, str]]:
    if not deck_context.deck_name:
        return []

    created_ids, errors = anki_port.apply_note_changes(
        deck_context.deck_name,
        deck_context.needs_create_deck,
        result.change_buckets.creates,
        result.change_buckets.updates,
        result.change_buckets.deletes,
        cards_to_move,
    )
    result.errors.extend(errors)
    delete_failed = any(err.startswith("Failed delete") for err in errors)

    note_key_assignments: list[tuple[Note, str]] = []

    # Link mapped created IDs
    for note_id, create_change in zip(created_ids, result.change_buckets.creates):
        note_key = create_change.context["note_key"]
        queue_note_mapping(note_key, note_id)
        queue_fingerprint(
            note_key,
            create_change.context["md_hash"],
            create_change.context["anki_hash"],
        )
        create_change.entity_id = note_id
        note_key_assignments.append((create_change.context["note"], note_key))

    for update_change in result.change_buckets.updates:
        queue_fingerprint(
            update_change.context["note_key"],
            update_change.context["md_hash"],
            update_change.context["anki_hash"],
        )

    if result.change_buckets.deletes and not delete_failed:
        delete_note_keys = [
            delete_change.context.get("note_key")
            for delete_change in result.change_buckets.deletes
            if delete_change.context
        ]
        delete_note_keys = [note_key for note_key in delete_note_keys if note_key]
        if delete_note_keys:
            db_port.remove_notes_by_note_keys_bulk(delete_note_keys)
            for note_key in delete_note_keys:
                removed_note_id = note_ids_by_note_key.pop(note_key, None)
                if note_key in global_note_keys and removed_note_id is not None:
                    global_mapped_note_ids.discard(removed_note_id)
                note_fingerprints_by_note_key.pop(note_key, None)

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
        db_port.set_deck(deck_context.deck_name, new_id)


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
    note_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_fingerprints: list[tuple[str, str, str]],
    global_note_keys: set[str],
    global_mapped_note_ids: set[int],
    all_anki_note_ids: set[int],
) -> tuple[SyncResult, _PendingWrite, set[str]]:
    deck_context = _resolve_deck_context(fs, deck_ids_by_name, db_port)
    result = SyncResult.for_notes(name=deck_context.deck_name, file_path=fs.file_path)
    cards_to_move: list[int] = []
    moved_from_decks: set[str] = set()

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

    def _queue_fingerprint(note_key: str, md_hash: str, anki_hash: str) -> None:
        if note_fingerprints_by_note_key.get(note_key) == (md_hash, anki_hash):
            return
        note_fingerprints_by_note_key[note_key] = (md_hash, anki_hash)
        pending_fingerprints.append((note_key, md_hash, anki_hash))

    for parsed_note in fs.notes:
        cfg = config_by_name[parsed_note.note_type]
        md_hash = note_fingerprint(parsed_note.note_type, parsed_note.fields)

        if parsed_note.note_key:
            _sync_keyed_note(
                parsed_note=parsed_note,
                cfg=cfg,
                md_hash=md_hash,
                deck_name=deck_context.deck_name,
                anki_port=anki_port,
                fs_port=fs_port,
                anki_notes=anki_notes,
                anki_cards=anki_cards,
                note_ids_by_note_key=note_ids_by_note_key,
                note_fingerprints_by_note_key=note_fingerprints_by_note_key,
                result=result,
                cards_to_move=cards_to_move,
                moved_from_decks=moved_from_decks,
                queue_note_mapping=_queue_note_mapping,
                queue_fingerprint=_queue_fingerprint,
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
        db_port=db_port,
        result=result,
    )

    note_key_assignments = _apply_changes_and_update_state(
        deck_context=deck_context,
        anki_port=anki_port,
        db_port=db_port,
        result=result,
        cards_to_move=cards_to_move,
        note_ids_by_note_key=note_ids_by_note_key,
        global_note_keys=global_note_keys,
        global_mapped_note_ids=global_mapped_note_ids,
        note_fingerprints_by_note_key=note_fingerprints_by_note_key,
        queue_note_mapping=_queue_note_mapping,
        queue_fingerprint=_queue_fingerprint,
    )

    _recover_created_deck_mapping(
        deck_context=deck_context,
        anki_port=anki_port,
        deck_ids_by_name=deck_ids_by_name,
        deck_names_by_id=deck_names_by_id,
        db_port=db_port,
    )

    result.materialize_changes()
    pending = _PendingWrite(fs, note_key_assignments)
    return result, pending, moved_from_decks


def import_collection(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    note_types_dir: Path,
) -> CollectionResult:
    configs = fs_port.load_note_type_configs(note_types_dir)
    config_by_name = {config.name: config for config in configs}
    md_files = fs_port.find_markdown_files(collection_dir)
    fs_docs = [
        fs_port.read_markdown_file(md_file, context_root=collection_dir)
        for md_file in md_files
    ]

    global_note_keys = set()
    note_key_sources = {}
    duplicates = []

    for markdown_file in fs_docs:
        for note in markdown_file.notes:
            if note.note_key:
                if note.note_key in note_key_sources:
                    duplicates.append(f"Duplicate note_key {note.note_key}")
                else:
                    note_key_sources[note.note_key] = markdown_file.file_path.name
                global_note_keys.add(note.note_key)

    if duplicates:
        raise ValueError(f"Aborting import: Duplicates found: {duplicates}")
    note_ids_by_note_key = db_port.get_note_ids_bulk(global_note_keys)
    note_fingerprints_by_note_key = db_port.get_note_fingerprints_bulk(global_note_keys)
    pending_note_mappings: list[tuple[str, int]] = []
    pending_fingerprints: list[tuple[str, str, str]] = []

    with db_port.transaction():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
        initial_deck_names = set(deck_ids_by_name)
        deck_names_by_id = {
            deck_id: deck_name for deck_name, deck_id in deck_ids_by_name.items()
        }

        all_note_ids = anki_port.fetch_all_note_ids([config.name for config in configs])
        anki_notes = anki_port.fetch_notes_info(all_note_ids)

        # Collaboration mode: rebuild note_key->note_id mappings from embedded
        # AnkiOps Key.
        # so move/delete decisions do not depend on a pre-existing local DB.
        for anki_note in anki_notes.values():
            embedded_note_key = anki_note.fields.get("AnkiOps Key", "").strip()
            if not embedded_note_key or embedded_note_key not in global_note_keys:
                continue
            if note_ids_by_note_key.get(embedded_note_key) == anki_note.note_id:
                continue
            note_ids_by_note_key[embedded_note_key] = anki_note.note_id
            pending_note_mappings.append((embedded_note_key, anki_note.note_id))

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

        note_ids_by_deck_name = {}
        for card_info in anki_cards.values():
            deck_name = card_info.get("deckName")
            note_id = card_info.get("note")
            if deck_name and note_id:
                note_ids_by_deck_name.setdefault(deck_name, set()).add(note_id)

        results = []
        pending = []
        md_deck_ids = set()
        markdown_deck_names = {
            file_stem_to_deck_name(markdown_file.file_path.stem)
            for markdown_file in fs_docs
        }
        renamed_from_decks: set[str] = set()

        for markdown_file in fs_docs:
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
                note_fingerprints_by_note_key,
                pending_fingerprints,
                global_note_keys,
                global_mapped_note_ids,
                file_anki_note_ids,
            )

            # Track filename-based deck renames to avoid stale DB deck mappings
            # and false-positive "untracked" warnings for the old deck.
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
                    renamed_from_decks.add(source_deck)
                    db_port.remove_deck(source_deck)
                    logger.info(
                        "Deck renamed from markdown file: "
                        f"'{source_deck}' -> '{sync_result.name}'"
                    )

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
            db_port.set_notes_bulk(pending_note_mappings)
        if pending_fingerprints:
            db_port.set_note_fingerprints_bulk(pending_fingerprints)
        db_port.prune_orphan_note_fingerprints()

        _flush_writes(fs_port, pending)
        db_port.save()

        untracked = []
        for deck_name, note_ids in note_ids_by_deck_name.items():
            if deck_name in renamed_from_decks:
                continue
            deck_id = deck_ids_by_name.get(deck_name)
            if deck_id is None or deck_id in md_deck_ids:
                continue
            untracked.append(UntrackedDeck(deck_name, deck_id, list(note_ids)))

    return CollectionResult.for_import(results=results, untracked_decks=untracked)

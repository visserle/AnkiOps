"""Use Case: Import Markdown Notes into Anki."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import NOTE_SEPARATOR
from ankiops.db import SQLiteDbAdapter
from ankiops.fingerprints import note_fingerprint
from ankiops.fs import FileSystemAdapter
from ankiops.models import (
    AnkiNote,
    Change,
    ChangeType,
    CollectionImportResult,
    MarkdownFile,
    Note,
    NoteSyncResult,
    NoteTypeConfig,
    UntrackedDeck,
)

logger = logging.getLogger(__name__)
_NOTE_KEY_LINE_RE = re.compile(r"^\s*<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->\s*$")


@dataclass
class _PendingWrite:
    file_state: MarkdownFile
    key_assignments: list[tuple[Note, str]]


def _find_first_markdown_line_index(note: Note, lines: list[str]) -> int:
    """Find the line index of the first field line within a note block."""
    first_value = note.first_field_line()
    if not first_value:
        return next((i for i, line in enumerate(lines) if line.strip()), 0)

    first_value_stripped = first_value.strip()
    if not first_value_stripped:
        return next((i for i, line in enumerate(lines) if line.strip()), 0)

    # Search the note block for a line containing this value with a prefix.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(first_value_stripped) and ":" in stripped:
            # Verify this is a prefixed line (e.g. "Q: Question 1").
            try:
                idx = stripped.rindex(first_value_stripped)
                prefix_part = stripped[:idx].rstrip()
                if prefix_part.endswith(":"):
                    return i
            except ValueError:
                pass

    return next((i for i, line in enumerate(lines) if line.strip()), 0)


def _upsert_note_key_in_block(block: str, note: Note, key_str: str) -> str:
    """Insert or replace note_key in a single markdown note block."""
    new_key = f"<!-- note_key: {key_str} -->"
    lines = block.split("\n")

    # Replace an existing note_key comment in place.
    for i, line in enumerate(lines):
        if _NOTE_KEY_LINE_RE.match(line):
            if line.strip() == new_key:
                return block
            lines[i] = new_key
            return "\n".join(lines)

    insert_idx = _find_first_markdown_line_index(note, lines)
    lines.insert(insert_idx, new_key)
    return "\n".join(lines)


def _flush_writes(fs_port: FileSystemAdapter, writes: list[_PendingWrite]) -> None:
    for w in writes:
        if not w.key_assignments:
            continue

        key_by_note = {id(note): key for note, key in w.key_assignments}
        blocks = w.file_state.raw_content.split(NOTE_SEPARATOR)
        out_blocks: list[str] = []
        note_idx = 0
        changed = False

        for block in blocks:
            if not block.strip() or set(block.strip()) <= {"-"}:
                out_blocks.append(block)
                continue

            if note_idx >= len(w.file_state.notes):
                out_blocks.append(block)
                continue

            note = w.file_state.notes[note_idx]
            note_idx += 1
            key_str = key_by_note.get(id(note))
            if key_str is None:
                out_blocks.append(block)
                continue

            updated_block = _upsert_note_key_in_block(block, note, key_str)
            changed = changed or updated_block != block
            out_blocks.append(updated_block)

        if note_idx != len(w.file_state.notes):
            raise ValueError(
                "Failed to align parsed notes with markdown blocks in "
                f"{w.file_state.file_path.name}"
            )

        if changed:
            fs_port.write_markdown_file(
                w.file_state.file_path, NOTE_SEPARATOR.join(out_blocks)
            )


def _to_html(
    note: Note, config: NoteTypeConfig, converter: FileSystemAdapter
) -> dict[str, str]:
    html = {
        name: converter.convert_to_html(content)
        for name, content in note.fields.items()
    }
    for f in config.fields:
        if f.prefix is None:
            continue
        html.setdefault(f.name, "")
    return html


def _html_match(html: dict[str, str], anki: AnkiNote) -> bool:
    return all(anki.fields.get(k) == v for k, v in html.items())


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
    note_ids_by_key: dict[str, int],
    pending_note_mappings: list[tuple[str, int]],
    note_fingerprints_by_key: dict[str, tuple[str, str]],
    pending_fingerprints: list[tuple[str, str, str]],
    global_keys: set[str],
    all_anki_note_ids: set[int],
) -> tuple[NoteSyncResult, _PendingWrite]:
    needs_create_deck = False

    deck_name = fs.file_path.stem.replace("__", "::")
    resolved_id = deck_ids_by_name.get(deck_name)

    if not resolved_id:
        needs_create_deck = True
    else:
        db_port.set_deck(deck_name, resolved_id)

    result = NoteSyncResult(deck_name=deck_name, file_path=fs.file_path)
    creates, updates, deletes, skips, moves = [], [], [], [], []
    cards_to_move = []

    def _queue_note_mapping(key: str, note_id: int) -> None:
        if note_ids_by_key.get(key) == note_id:
            return
        note_ids_by_key[key] = note_id
        pending_note_mappings.append((key, note_id))

    def _queue_fingerprint(key: str, md_hash: str, anki_hash: str) -> None:
        if note_fingerprints_by_key.get(key) == (md_hash, anki_hash):
            return
        note_fingerprints_by_key[key] = (md_hash, anki_hash)
        pending_fingerprints.append((key, md_hash, anki_hash))

    for parsed_note in fs.notes:
        cfg = config_by_name[parsed_note.note_type]
        md_hash = note_fingerprint(parsed_note.note_type, parsed_note.fields)

        if parsed_note.note_key:
            note_key = parsed_note.note_key
            note_id = note_ids_by_key.get(note_key)
            if note_id is None:
                # Recovery
                found = anki_port.find_notes_by_ankiops_key(note_key)
                if found:
                    note_id = found[0]
                    _queue_note_mapping(note_key, note_id)

            anki_note = anki_notes.get(note_id) if note_id else None

            # If recovery worked but wasn't in original batch fetch
            if note_id and not anki_note:
                info = anki_port.fetch_notes_info([note_id])
                if info:
                    anki_note = info[note_id]
                    anki_notes[note_id] = anki_note
                else:
                    # Stale key->id mapping: recover via embedded AnkiOps Key.
                    found = anki_port.find_notes_by_ankiops_key(note_key)
                    if found:
                        note_id = found[0]
                        _queue_note_mapping(note_key, note_id)
                        info = anki_port.fetch_notes_info([note_id])
                        if info:
                            anki_note = info[note_id]
                            anki_notes[note_id] = anki_note

            if not anki_note:
                html_fields = _to_html(parsed_note, cfg, fs_port)
                html_fields["AnkiOps Key"] = note_key
                creates.append(
                    Change(
                        ChangeType.CREATE,
                        None,
                        parsed_note.identifier,
                        {
                            "note": parsed_note,
                            "html_fields": html_fields,
                            "note_key": note_key,
                            "md_hash": md_hash,
                            "anki_hash": note_fingerprint(
                                parsed_note.note_type, html_fields
                            ),
                        },
                    )
                )
                logger.debug(f"  Create {parsed_note.identifier}")
            else:
                c_to_move = []
                for cid in anki_note.card_ids:
                    c = anki_cards.get(cid)
                    if c and c.get("deckName") != deck_name:
                        c_to_move.append(cid)
                if c_to_move:
                    moves.append(
                        Change(
                            ChangeType.MOVE,
                            note_id,
                            parsed_note.identifier,
                            {"cards": c_to_move},
                        )
                    )
                    cards_to_move.extend(c_to_move)

                if anki_note.note_type != parsed_note.note_type:
                    result.errors.append(
                        f"Note type mismatch for {parsed_note.identifier}"
                    )
                    continue

                current_anki_hash = note_fingerprint(
                    anki_note.note_type, anki_note.fields
                )
                cached = note_fingerprints_by_key.get(note_key)
                if cached == (md_hash, current_anki_hash):
                    skips.append(
                        Change(ChangeType.SKIP, note_id, parsed_note.identifier)
                    )
                    _queue_fingerprint(note_key, md_hash, current_anki_hash)
                    continue

                html_fields = _to_html(parsed_note, cfg, fs_port)
                if anki_note.fields.get("AnkiOps Key", "") != note_key:
                    html_fields["AnkiOps Key"] = note_key

                if _html_match(html_fields, anki_note):
                    skips.append(
                        Change(ChangeType.SKIP, note_id, parsed_note.identifier)
                    )
                    _queue_fingerprint(note_key, md_hash, current_anki_hash)
                else:
                    target_anki_hash = note_fingerprint(
                        parsed_note.note_type, html_fields
                    )
                    updates.append(
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
        else:
            new_key = db_port.generate_key()
            html_fields = _to_html(parsed_note, cfg, fs_port)
            html_fields["AnkiOps Key"] = new_key
            creates.append(
                Change(
                    ChangeType.CREATE,
                    None,
                    parsed_note.identifier,
                    {
                        "note": parsed_note,
                        "html_fields": html_fields,
                        "note_key": new_key,
                        "md_hash": md_hash,
                        "anki_hash": note_fingerprint(
                            parsed_note.note_type, html_fields
                        ),
                    },
                )
            )
            logger.debug(f"  Create (new) {parsed_note.identifier}")

    md_anki_ids = set()
    for n in fs.notes:
        if n.note_key:
            nid = note_ids_by_key.get(n.note_key)
            if nid:
                md_anki_ids.add(nid)

    # Orphans
    orphaned = all_anki_note_ids - md_anki_ids
    if global_keys:
        g_ids = set()
        for key in global_keys:
            mapped = note_ids_by_key.get(key)
            if mapped:
                g_ids.add(mapped)
        orphaned -= g_ids

    for nid in orphaned:
        if nid:
            key = db_port.get_note_key(nid)
            deletes.append(
                Change(
                    ChangeType.DELETE,
                    nid,
                    f"note_key: {key}" if key else f"note_id: {nid}",
                    {"note_key": key} if key else {},
                )
            )
            logger.debug(
                f"  Delete {f'note_key: {key}' if key else f'note_id: {nid}'}"
            )

    key_assignments = []
    if deck_name:
        created_ids, errors = anki_port.apply_note_changes(
            deck_name, needs_create_deck, creates, updates, deletes, cards_to_move
        )
        result.errors.extend(errors)
        delete_failed = any(err.startswith("Failed delete") for err in errors)

        # Link mapped created IDs
        for cid, c in zip(created_ids, creates):
            k = c.context["note_key"]
            _queue_note_mapping(k, cid)
            _queue_fingerprint(k, c.context["md_hash"], c.context["anki_hash"])
            c.entity_id = cid
            key_assignments.append((c.context["note"], k))

        for c in updates:
            _queue_fingerprint(
                c.context["note_key"], c.context["md_hash"], c.context["anki_hash"]
            )

        if deletes and not delete_failed:
            delete_keys = [c.context.get("note_key") for c in deletes if c.context]
            delete_keys = [k for k in delete_keys if k]
            if delete_keys:
                db_port.remove_notes_by_keys_bulk(delete_keys)
                for key in delete_keys:
                    note_ids_by_key.pop(key, None)
                    note_fingerprints_by_key.pop(key, None)

        # Deck recovery logic for needs_create_deck
        if needs_create_deck:
            new_deck_ids = anki_port.fetch_deck_names_and_ids()
            if deck_name in new_deck_ids:
                new_id = new_deck_ids[deck_name]
                deck_ids_by_name[deck_name] = new_id
                deck_names_by_id[new_id] = deck_name
                db_port.set_deck(deck_name, new_id)

    result.changes = creates + updates + deletes + skips + moves
    pending = _PendingWrite(fs, key_assignments)
    return result, pending


def import_collection(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    note_types_dir: Path,
) -> CollectionImportResult:
    configs = fs_port.load_note_type_configs(note_types_dir)
    config_by_name = {c.name: c for c in configs}
    md_files = fs_port.find_markdown_files(collection_dir)

    fs_docs = [fs_port.read_markdown_file(f) for f in md_files]

    global_keys = set()
    key_sources = {}
    duplicates = []

    for fs in fs_docs:
        for n in fs.notes:
            if n.note_key:
                if n.note_key in key_sources:
                    duplicates.append(f"Duplicate note_key {n.note_key}")
                else:
                    key_sources[n.note_key] = fs.file_path.name
                global_keys.add(n.note_key)

    if duplicates:
        raise ValueError(f"Aborting import: Duplicates found: {duplicates}")
    note_ids_by_key = db_port.get_note_ids_bulk(global_keys)
    note_fingerprints_by_key = db_port.get_note_fingerprints_bulk(global_keys)
    pending_note_mappings: list[tuple[str, int]] = []
    pending_fingerprints: list[tuple[str, str, str]] = []

    with db_port.transaction():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
        deck_names_by_id = {v: k for k, v in deck_ids_by_name.items()}

        all_note_ids = anki_port.fetch_all_note_ids([c.name for c in configs])
        anki_notes = anki_port.fetch_notes_info(all_note_ids)

        # Collaboration mode: rebuild key->id mappings from embedded AnkiOps Key
        # so move/delete decisions do not depend on a pre-existing local DB.
        for anki_note in anki_notes.values():
            embedded_key = anki_note.fields.get("AnkiOps Key", "").strip()
            if not embedded_key or embedded_key not in global_keys:
                continue
            if note_ids_by_key.get(embedded_key) == anki_note.note_id:
                continue
            note_ids_by_key[embedded_key] = anki_note.note_id
            pending_note_mappings.append((embedded_key, anki_note.note_id))

        # We need to compute anki_cards and group note_ids by deck.
        # A single bulk cardsInfo fetch is faster than many smaller calls.
        all_card_ids = []
        for n in anki_notes.values():
            all_card_ids.extend(n.card_ids)
        anki_cards = anki_port.fetch_cards_info(all_card_ids)

        note_ids_by_deck_name = {}
        for cid, c in anki_cards.items():
            dname = c.get("deckName")
            nid = c.get("note")
            if dname and nid:
                note_ids_by_deck_name.setdefault(dname, set()).add(nid)

        results = []
        pending = []
        md_deck_ids = set()

        for fs in fs_docs:
            # Determine actual anki_deck_note_ids for this specific file
            # We find what the deck name maps to...
            dn = fs.file_path.stem.replace("__", "::")

            file_anki_note_ids = note_ids_by_deck_name.get(dn, set())

            res, p = _sync_file(
                fs,
                config_by_name,
                anki_port,
                db_port,
                fs_port,
                deck_names_by_id,
                deck_ids_by_name,
                anki_notes,
                anki_cards,
                note_ids_by_key,
                pending_note_mappings,
                note_fingerprints_by_key,
                pending_fingerprints,
                global_keys,
                file_anki_note_ids,
            )
            if res.deck_name in deck_ids_by_name:
                md_deck_ids.add(deck_ids_by_name[res.deck_name])

            results.append(res)
            pending.append(p)
            summary = res.summary
            if summary.format() != "no changes":
                logger.debug(f"File '{fs.file_path.name}': {summary.format()}")

        if pending_note_mappings:
            db_port.set_notes_bulk(pending_note_mappings)
        if pending_fingerprints:
            db_port.set_note_fingerprints_bulk(pending_fingerprints)

        _flush_writes(fs_port, pending)
        db_port.save()

        untracked = []
        for dname, nids in note_ids_by_deck_name.items():
            did = deck_ids_by_name.get(dname)
            if did is None or did in md_deck_ids:
                continue
            untracked.append(UntrackedDeck(dname, did, list(nids)))

    return CollectionImportResult(results, untracked)

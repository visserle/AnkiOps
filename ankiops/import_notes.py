"""Use Case: Import Markdown Notes into Anki."""

import logging
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.db import SQLiteDbAdapter
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


@dataclass
class _PendingWrite:
    file_state: MarkdownFile
    key_assignments: list[tuple[Note, str]]


def _build_first_markdown_line(note: Note, fs: MarkdownFile) -> str:
    """Find the full markdown line (prefix + value) for the first field of a note.

    We search the raw file content for the first field's prefixed line.
    This is needed because first_field_line() returns just the field value,
    which could match inside a prefixed line and corrupt the file.
    """
    first_value = note.first_field_line()
    if not first_value:
        return ""

    # Search raw content for a line containing this value with a prefix
    for line in fs.raw_content.split("\n"):
        stripped = line.strip()
        if stripped.endswith(first_value) and ":" in stripped:
            # Verify it's a field prefix line (e.g., "Q: Question 1")
            prefix_part = stripped[: stripped.index(first_value)].rstrip()
            if prefix_part.endswith(":"):
                return stripped
    # Fallback to raw first line
    return first_value


def _flush_writes(fs_port: FileSystemAdapter, writes: list[_PendingWrite]) -> None:
    for w in writes:
        if not w.key_assignments:
            continue

        content = w.file_state.raw_content

        if w.key_assignments:
            # Check for duplicates on first line to ensure safe insertion
            first_lines = {}
            for n, _ in w.key_assignments:
                fl = n.first_field_line()
                first_lines.setdefault(fl, []).append(n.identifier)
            dupes = {k: v for k, v in first_lines.items() if len(v) > 1}
            if dupes:
                raise ValueError(
                    f"Duplicate first lines prevent key assignment in {w.file_state.file_path.name}"
                )

            for note, key_str in w.key_assignments:
                new_key = f"<!-- note_key: {key_str} -->"
                if note.note_key:
                    old_key = f"<!-- note_key: {note.note_key} -->"
                    content = content.replace(old_key, new_key, 1)
                else:
                    # Build the full markdown line (prefix + value) to find in the raw content
                    # Using just the field value would match inside a prefixed line and corrupt it
                    first_line = _build_first_markdown_line(note, w.file_state)
                    content = content.replace(
                        first_line, new_key + "\n" + first_line, 1
                    )

        fs_port.write_markdown_file(w.file_state.file_path, content)


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
    configs: list[NoteTypeConfig],
    anki_port: AnkiAdapter,
    db_port: SQLiteDbAdapter,
    fs_port: FileSystemAdapter,
    deck_names_by_id: dict[int, str],
    deck_ids_by_name: dict[str, int],
    anki_notes: dict[int, AnkiNote],
    anki_cards: dict[int, dict],
    only_add_new: bool,
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

    for parsed_note in fs.notes:
        cfg = next(c for c in configs if c.name == parsed_note.note_type)
        html_fields = _to_html(parsed_note, cfg, fs_port)

        if parsed_note.note_key:
            note_id = db_port.get_note_id(parsed_note.note_key)
            if note_id is None:
                # Recovery
                found = anki_port.find_notes_by_ankiops_key(parsed_note.note_key)
                if found:
                    note_id = found[0]
                    db_port.set_note(parsed_note.note_key, note_id)

            if note_id and only_add_new:
                skips.append(Change(ChangeType.SKIP, note_id, parsed_note.identifier))
                continue

            anki_note = anki_notes.get(note_id) if note_id else None

            # If recovery worked but wasn't in original batch fetch
            if note_id and not anki_note:
                info = anki_port.fetch_notes_info([note_id])
                if info:
                    anki_note = info[note_id]
                    anki_notes[note_id] = anki_note

            if not anki_note:
                creates.append(
                    Change(
                        ChangeType.CREATE,
                        None,
                        parsed_note.identifier,
                        {
                            "note": parsed_note,
                            "html_fields": html_fields,
                            "note_key": parsed_note.note_key,
                        },
                    )
                )
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

                if anki_note.fields.get("AnkiOps Key", "") != parsed_note.note_key:
                    html_fields["AnkiOps Key"] = parsed_note.note_key

                if _html_match(html_fields, anki_note):
                    skips.append(
                        Change(ChangeType.SKIP, note_id, parsed_note.identifier)
                    )
                else:
                    updates.append(
                        Change(
                            ChangeType.UPDATE,
                            note_id,
                            parsed_note.identifier,
                            {"html_fields": html_fields},
                        )
                    )
        else:
            new_key = db_port.generate_key()
            creates.append(
                Change(
                    ChangeType.CREATE,
                    None,
                    parsed_note.identifier,
                    {
                        "note": parsed_note,
                        "html_fields": html_fields,
                        "note_key": new_key,
                    },
                )
            )

    md_anki_ids = set()
    for n in fs.notes:
        if n.note_key:
            nid = db_port.get_note_id(n.note_key)
            if nid:
                md_anki_ids.add(nid)

    # Orphans
    orphaned = all_anki_note_ids - md_anki_ids
    if global_keys:
        g_ids = {db_port.get_note_id(k) for k in global_keys if db_port.get_note_id(k)}
        orphaned -= g_ids

    for nid in orphaned:
        if nid:
            key = db_port.get_note_key(nid)
            deletes.append(
                Change(
                    ChangeType.DELETE,
                    nid,
                    f"note_key: {key}" if key else f"note_id: {nid}",
                )
            )

    key_assignments = []
    if deck_name:
        created_ids, errors = anki_port.apply_note_changes(
            deck_name, needs_create_deck, creates, updates, deletes, cards_to_move
        )
        result.errors.extend(errors)

        # Link mapped created IDs
        for cid, c in zip(created_ids, creates):
            k = c.context["note_key"]
            db_port.set_note(k, cid)
            c.entity_id = cid
            key_assignments.append((c.context["note"], k))

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
    only_add_new: bool = False,
) -> CollectionImportResult:
    configs = fs_port.load_note_type_configs(note_types_dir)
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

    deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
    deck_names_by_id = {v: k for k, v in deck_ids_by_name.items()}

    all_note_ids = anki_port.fetch_all_note_ids([c.name for c in configs])
    anki_notes = anki_port.fetch_notes_info(all_note_ids)

    # We need to compute anki_cards and group note_ids by deck
    # For performance in the UseCase, doing one massive fetch of cards info is faster.
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
            configs,
            anki_port,
            db_port,
            fs_port,
            deck_names_by_id,
            deck_ids_by_name,
            anki_notes,
            anki_cards,
            only_add_new,
            global_keys,
            file_anki_note_ids,
        )
        if res.deck_name in deck_ids_by_name:
            md_deck_ids.add(deck_ids_by_name[res.deck_name])

        results.append(res)
        pending.append(p)

    _flush_writes(fs_port, pending)
    db_port.save()

    untracked = []
    for dname, nids in note_ids_by_deck_name.items():
        did = deck_ids_by_name.get(dname)
        if did is None or did in md_deck_ids:
            continue
        untracked.append(UntrackedDeck(dname, did, list(nids)))

    return CollectionImportResult(results, untracked)

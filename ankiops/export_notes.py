"""Use Case: Export Anki Notes to Markdown."""

import logging
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import NOTE_SEPARATOR
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.models import (
    AnkiNote,
    Change,
    ChangeType,
    CollectionExportResult,
    Note,
    NoteSyncResult,
    NoteTypeConfig,
)

logger = logging.getLogger(__name__)


def _from_html(
    anki_note: AnkiNote,
    config: NoteTypeConfig,
    fs_port: FileSystemAdapter,
) -> Note:
    """Convert an AnkiNote into a Domain Note using the FS Port."""
    fields = {}
    for f in config.fields:
        if f.name == "AnkiOps Key":
            continue
        if f.name in anki_note.fields:
            md_val = fs_port.convert_to_markdown(anki_note.fields[f.name])
            if md_val:
                fields[f.name] = md_val

    note_key = anki_note.fields.get("AnkiOps Key", "").strip()
    return Note(
        note_key=note_key if note_key else None,
        note_type=anki_note.note_type,
        fields=fields,
    )


def _sync_deck(
    deck_name: str,
    deck_id: int,
    anki_notes: list[AnkiNote],
    configs: list[NoteTypeConfig],
    existing_file_path: Path | None,
    collection_dir: Path,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
) -> NoteSyncResult:
    result = NoteSyncResult(deck_name=deck_name, file_path=existing_file_path)

    if existing_file_path and existing_file_path.exists():
        fs = fs_port.read_markdown_file(existing_file_path)
    else:
        file_path = collection_dir / f"{deck_name.replace('::', '__')}.md"
        result.file_path = file_path

        content = ""
        fs_port.write_markdown_file(file_path, content)
        fs = fs_port.read_markdown_file(file_path)

        # Sort by note_id (creation timestamp) for first-time exports
        anki_notes = sorted(anki_notes, key=lambda n: n.note_id)

    local_notes_by_key = {n.note_key: n for n in fs.notes if n.note_key}
    local_notes_by_content = {
        n.first_field_line(): n for n in fs.notes if not n.note_key
    }

    creates, updates, skips = [], [], []
    final_notes = []

    for anki_note in anki_notes:
        cfg = next((c for c in configs if c.name == anki_note.note_type), None)
        if not cfg:
            result.errors.append(
                f"Unknown note type {anki_note.note_type} for note {anki_note.note_id}"
            )
            continue

        domain_note = _from_html(anki_note, cfg, fs_port)
        key = db_port.get_note_key(anki_note.note_id)

        if not key:
            # Maybe it has an embedded key
            embedded = anki_note.fields.get("AnkiOps Key", "").strip()
            if embedded:
                db_port.set_note(embedded, anki_note.note_id)
                key = embedded
            else:
                # Match by content
                first_line = domain_note.first_field_line()
                match = local_notes_by_content.get(first_line)
                if match and match.note_key:
                    key = match.note_key
                    db_port.set_note(key, anki_note.note_id)
                else:
                    key = db_port.generate_key()
                    db_port.set_note(key, anki_note.note_id)

        domain_note.note_key = key
        local_match = local_notes_by_key.get(key)

        if not local_match:
            creates.append(
                Change(ChangeType.CREATE, anki_note.note_id, domain_note.identifier)
            )
            logger.debug(f"  Created {domain_note.identifier}")
            final_notes.append(domain_note)
        else:
            if local_match.fields == domain_note.fields:
                skips.append(
                    Change(ChangeType.SKIP, anki_note.note_id, domain_note.identifier)
                )
                final_notes.append(local_match)
            else:
                updates.append(
                    Change(ChangeType.UPDATE, anki_note.note_id, domain_note.identifier)
                )
                logger.debug(f"  Updated {domain_note.identifier}")
                final_notes.append(domain_note)

    # Rebuild file content
    content_parts = []

    for note in final_notes:
        parts = []
        if note.note_key:
            parts.append(f"<!-- note_key: {note.note_key} -->")

        cfg = next(c for c in configs if c.name == note.note_type)
        for f in cfg.fields:
            if f.prefix and f.name in note.fields and note.fields[f.name]:
                lines = note.fields[f.name].split("\n")
                parts.append(f"{f.prefix} {lines[0]}")
                if len(lines) > 1:
                    parts.extend(lines[1:])

        content_parts.append("\n".join(parts))

    final_text = NOTE_SEPARATOR.join(content_parts) + "\n"

    if final_text.strip() != fs.raw_content.strip():
        fs_port.write_markdown_file(result.file_path, final_text)

    # Detect deleted notes
    final_keys = {n.note_key for n in final_notes if n.note_key}
    for old_note in fs.notes:
        if old_note.note_key and old_note.note_key not in final_keys:
            # Note was deleted in Anki
            db_port.remove_note_by_key(old_note.note_key)
            result.changes.append(Change(ChangeType.DELETE, None, old_note.identifier))
            logger.debug(f"  Deleted {old_note.identifier}")

    result.changes.extend(creates + updates + skips)
    return result


def export_collection(
    anki_port: AnkiAdapter,
    fs_port: FileSystemAdapter,
    db_port: SQLiteDbAdapter,
    collection_dir: Path,
    note_types_dir: Path,
    keep_orphans: bool = False,
) -> CollectionExportResult:
    configs = fs_port.load_note_type_configs(note_types_dir)

    deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
    deck_names_by_id = {v: k for k, v in deck_ids_by_name.items()}

    all_note_ids = anki_port.fetch_all_note_ids([c.name for c in configs])
    anki_notes = anki_port.fetch_notes_info(all_note_ids)

    # Fetch cards info to map notes to decks
    all_card_ids = []
    for n in anki_notes.values():
        all_card_ids.extend(n.card_ids)
    anki_cards = anki_port.fetch_cards_info(all_card_ids)

    notes_by_deck = {}
    for note_id, anki_note in anki_notes.items():
        if not anki_note.card_ids:
            continue
        c = anki_cards.get(anki_note.card_ids[0])
        if not c:
            continue
        dname = c.get("deckName")
        notes_by_deck.setdefault(dname, []).append(anki_note)

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
        safe_name = deck_name.replace("::", "__")
        if old_name and old_name != deck_name:
            old_safe = old_name.replace("::", "__")
            if old_safe in file_map_by_name:
                old_path = file_map_by_name[old_safe]
                new_path = old_path.parent / f"{safe_name}.md"
                old_path.rename(new_path)
                file_map_by_name[safe_name] = new_path
                del file_map_by_name[old_safe]
                logger.info(f"Deck renamed: '{old_name}' → '{deck_name}'")

        target_file = file_map_by_name.get(safe_name)

        res = _sync_deck(
            deck_name,
            deck_id,
            notes,
            configs,
            target_file,
            collection_dir,
            fs_port,
            db_port,
        )
        db_port.set_deck(deck_name, deck_id)
        results.append(res)
        summary = res.summary
        if summary.format() != "no changes":
            logger.debug(f"Deck '{deck_name}': {summary.format()}")

    extra_changes = []
    if not keep_orphans:
        active_files = {res.file_path for res in results if res.file_path}
        for md_file in md_files:
            if md_file not in active_files:
                db_port.remove_deck(md_file.stem.replace("__", "::"))
                md_file.unlink()
                extra_changes.append(
                    Change(ChangeType.DELETE, None, f"file: {md_file.name}")
                )

    db_port.save()
    return CollectionExportResult(results, extra_changes)

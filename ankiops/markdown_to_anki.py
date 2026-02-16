"""Import Markdown files back into Anki.

Architecture:
  AnkiState   – all Anki-side data, fetched once (shared from anki_client)
  FileState   – one markdown file, read once (from models)
  _sync_file  – single engine: classify → update → delete → create
  _flush_writes – deferred file I/O (one write per file, at the end)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki_client import invoke
from ankiops.log import clickable_path, format_changes
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import (
    AnkiState,
    FileState,
    ImportSummary,
    InvalidID,
    Note,
    SyncResult,
    UntrackedDeck,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingWrite:
    """Deferred file modification, applied after all Anki mutations."""

    file_state: FileState
    deck_id_to_write: int | None
    id_assignments: list[tuple[Note, int]]



def _prompt_invalid_ids(invalid_ids: list[InvalidID], is_collection: bool) -> None:
    """Prompt user about invalid IDs and exit if they decline to continue.

    Args:
        invalid_ids: List of IDs that don't exist in Anki
        is_collection: True if importing a collection (plural), False for single file
    """
    if not invalid_ids:
        return

    deck_ids = [x for x in invalid_ids if x.id_type == "deck_id"]
    note_ids = [x for x in invalid_ids if x.id_type == "note_id"]

    file_text = "markdown files" if is_collection else "the markdown file"
    logger.warning(
        f"Found IDs in {file_text} that do not exist in your Anki collection:"
    )

    if deck_ids:
        logger.warning(f"\n  Deck IDs ({len(deck_ids)}):")
        for inv_id in deck_ids[:5]:
            if is_collection:
                clickable = clickable_path(inv_id.file_path)
                logger.warning(f"    - {inv_id.id_value} in {clickable}")
            else:
                logger.warning(f"    - {inv_id.id_value}")
        if len(deck_ids) > 5:
            logger.warning(f"    ... and {len(deck_ids) - 5} more")

    if note_ids:
        logger.warning(f"\n  Note IDs ({len(note_ids)}):")
        for inv_id in note_ids[:5]:
            # Replace filename in context with clickable link
            clickable = clickable_path(inv_id.file_path)
            context = inv_id.context.replace(inv_id.file_path.name, clickable)
            logger.warning(f"    - {inv_id.id_value} ({context})")
        if len(note_ids) > 5:
            logger.warning(f"    ... and {len(note_ids) - 5} more")

    logger.warning("\nThese IDs cannot be used (they don't exist in Anki).")
    logger.warning(
        "New items will be created in Anki with new IDs, "
        "and your markdown files will be updated."
    )

    answer = input("\nContinue with import? [y/N] ").strip().lower()
    if answer != "y":
        logger.info("Import cancelled.")
        raise SystemExit(0)


def _flush_writes(writes: list[_PendingWrite]) -> None:
    """Apply all deferred file modifications (one write per file)."""
    for w in writes:
        if w.deck_id_to_write is None and not w.id_assignments:
            continue

        content = w.file_state.raw_content

        # 1. Insert or replace deck_id
        if w.deck_id_to_write is not None:
            _, remaining = FileState.extract_deck_id(content)
            content = f"<!-- deck_id: {w.deck_id_to_write} -->\n" + remaining

        # 2. Insert note_ids for new / stale notes
        if w.id_assignments:
            w.file_state.validate_no_duplicate_first_lines(w.id_assignments)

            for note, id_value in w.id_assignments:
                new_id_comment = f"<!-- note_id: {id_value} -->"
                if note.note_id is not None:
                    old_id_comment = f"<!-- note_id: {note.note_id} -->"
                    content = content.replace(old_id_comment, new_id_comment, 1)
                else:
                    content = content.replace(
                        note.first_line, new_id_comment + "\n" + note.first_line, 1
                    )

        w.file_state.file_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


def _sync_file(
    fs: FileState,
    anki: AnkiState,
    converter: MarkdownToHTML,
    only_add_new: bool = False,
    global_note_ids: set[int] | None = None,
) -> tuple[SyncResult, _PendingWrite]:
    """Synchronize one markdown file to Anki.

    Phases:
      0. Resolve deck (by deck_id or filename)
      1. Classify notes (existing vs new) + convert md → html
      2. Build unified action list (createDeck, changeDeck, updates, deletes, creates)
      3. Execute single multi call + route results
      4. Return result + pending file write

    All Anki mutations are batched into a single ``invoke("multi", ...)``
    call.  AnkiConnect executes multi-actions sequentially, so ordering
    (e.g. createDeck before addNote) is preserved.
    """
    # ---- Phase 0: Resolve deck ----
    deck_id_to_write: int | None = None
    needs_create_deck = False

    if fs.deck_id and fs.deck_id in anki.deck_names_by_id:
        deck_name = anki.deck_names_by_id[fs.deck_id]
        logger.debug(f"Resolved deck by ID {fs.deck_id}: {deck_name}")
    else:
        deck_name = fs.file_path.stem.replace("__", "::")

        if deck_name not in anki.deck_ids_by_name:
            needs_create_deck = True
        elif not fs.deck_id:
            deck_id_to_write = anki.deck_ids_by_name[deck_name]
            logger.debug(f"Wrote deck_id {deck_id_to_write} to {fs.file_path.name}")

    result = SyncResult(
        deck_name=deck_name,
        file_path=fs.file_path,
        note_count=len(fs.parsed_notes),
        updated_count=0,
        created_count=0,
        deleted_count=0,
        moved_count=0,
        skipped_count=0,
    )

    # ---- Phase 1: Classify + convert ----
    existing: list[tuple[Note, dict[str, str]]] = []
    new: list[tuple[Note, dict[str, str]]] = []

    for note in fs.parsed_notes:
        validation_errors = note.validate()
        if validation_errors:
            for err in validation_errors:
                result.errors.append(
                    f"Note {note.note_id or 'new'} ({note.identifier}): {err}"
                )
            continue

        try:
            html_fields = note.to_html(converter)
        except Exception as e:
            result.errors.append(
                f"Note {note.note_id or 'new'} ({note.identifier}): {e}"
            )
            continue

        if note.note_id:
            if only_add_new:
                result.skipped_count += 1
            else:
                existing.append((note, html_fields))
        else:
            new.append((note, html_fields))

    # ---- Phase 2: Build unified action list ----
    actions: list[dict] = []
    tags: list[str] = []  # one tag per action for result routing

    # -- createDeck (must come first so the deck exists for addNote) --
    if needs_create_deck:
        actions.append({"action": "createDeck", "params": {"deck": deck_name}})
        tags.append("create_deck")

    # -- Safety check: note type mismatches --
    for note, _ in existing:
        anki_note = anki.notes_by_id.get(note.note_id)  # type: ignore[arg-type]
        if anki_note and anki_note.note_type != note.note_type:
            raise ValueError(
                f"Note type mismatch for note {note.note_id} "
                f"({note.identifier}): "
                f"Markdown specifies '{note.note_type}' "
                f"but Anki has '{anki_note.note_type}'. "
                f"AnkiConnect does not support changing note types. "
                f"Please manually change the note type in Anki "
                f"or delete the old note_id HTML tag to re-create the note."
            )

    # -- changeDeck --
    cards_to_move: list[int] = []
    for note, _ in existing:
        anki_note = anki.notes_by_id.get(note.note_id)  # type: ignore[arg-type]
        if not anki_note:
            continue
        for cid in anki_note.card_ids:
            card = anki.cards_by_id.get(cid)
            if card and card.get("deckName") != deck_name:
                cards_to_move.append(cid)

    if cards_to_move:
        actions.append(
            {"action": "changeDeck", "params": {"cards": cards_to_move, "deck": deck_name}}
        )
        tags.append("change_deck")

    # -- updateNoteFields --
    stale: list[tuple[Note, dict[str, str]]] = []
    update_notes: list[Note] = []

    for note, html_fields in existing:
        anki_note = anki.notes_by_id.get(note.note_id)  # type: ignore[arg-type]
        if not anki_note or not anki_note.fields:
            logger.debug(
                f"Note {note.note_id} ({note.identifier}) "
                f"no longer in Anki, will re-create"
            )
            stale.append((note, html_fields))
            continue

        if note.html_fields_match(html_fields, anki_note):
            result.skipped_count += 1
            continue

        actions.append(
            {
                "action": "updateNoteFields",
                "params": {"note": {"id": note.note_id, "fields": html_fields}},
            }
        )
        update_notes.append(note)
        tags.append("update")

    new.extend(stale)

    # -- deleteNotes --
    md_note_ids = {n.note_id for n in fs.parsed_notes if n.note_id is not None}
    anki_deck_note_ids = anki.note_ids_by_deck_name.get(deck_name, set()).copy()
    orphaned = anki_deck_note_ids - md_note_ids
    if global_note_ids:
        orphaned -= global_note_ids

    if orphaned:
        for nid in orphaned:
            anki_note = anki.notes_by_id.get(nid)
            model = anki_note.note_type if anki_note else "unknown"
            cids = [
                cid
                for cid, c in anki.cards_by_id.items()
                if c["note"] == nid and c["deckName"] == deck_name
            ]
            cid_str = ", ".join(str(c) for c in cids)
            logger.debug(
                f"Deleted {model} note {nid}"
                f"{f' (cards {cid_str})' if cid_str else ''}"
                f" from Anki"
            )
        actions.append({"action": "deleteNotes", "params": {"notes": list(orphaned)}})
        tags.append("delete")

    # -- addNote --
    create_notes: list[Note] = []
    for note, html_fields in new:
        actions.append(
            {
                "action": "addNote",
                "params": {
                    "note": {
                        "deckName": deck_name,
                        "modelName": note.note_type,
                        "fields": html_fields,
                        "options": {"allowDuplicate": False},
                    }
                },
            }
        )
        create_notes.append(note)
        tags.append("create")

    # ---- Phase 3: Execute single multi call + route results ----
    id_assignments: list[tuple[Note, int]] = []

    if actions:
        logger.debug(f"Sending {len(actions)} batched actions for '{deck_name}'")
        try:
            multi_results = invoke("multi", actions=actions)

            update_idx = 0
            create_idx = 0
            for tag, res in zip(tags, multi_results):
                if tag == "create_deck":
                    new_deck_id = res
                    anki.deck_ids_by_name[deck_name] = new_deck_id
                    anki.deck_names_by_id[new_deck_id] = deck_name
                    deck_id_to_write = new_deck_id
                    logger.info(
                        f"Created new deck '{deck_name}' (id: {new_deck_id})"
                    )

                elif tag == "change_deck":
                    if res is not None:
                        result.errors.append(f"Failed to move cards: {res}")
                    else:
                        moved_notes: set[int] = set()
                        for cid in cards_to_move:
                            card = anki.cards_by_id.get(cid)
                            if card:
                                nid = card["note"]
                                if nid not in moved_notes:
                                    moved_notes.add(nid)
                                    logger.debug(
                                        f"Moved note {nid} from "
                                        f"'{card['deckName']}' to '{deck_name}'"
                                    )
                        result.moved_count += len(moved_notes)

                elif tag == "update":
                    note = update_notes[update_idx]
                    update_idx += 1
                    if res is None:
                        result.updated_count += 1
                    else:
                        result.errors.append(
                            f"Note {note.note_id} ({note.identifier}): {res}"
                        )

                elif tag == "delete":
                    if res is not None:
                        result.errors.append(f"Failed to delete orphaned notes: {res}")
                    else:
                        result.deleted_count += len(orphaned)

                elif tag == "create":
                    note = create_notes[create_idx]
                    create_idx += 1
                    if res and isinstance(res, int):
                        id_assignments.append((note, res))
                        result.created_count += 1
                    else:
                        result.errors.append(
                            f"Note new ({note.identifier}): {res}"
                        )

        except Exception as e:
            # The entire multi call failed — attribute errors to all actions
            for note in update_notes:
                result.errors.append(f"Note {note.note_id} ({note.identifier}): {e}")
            for note in create_notes:
                result.errors.append(f"Note new ({note.identifier}): {e}")
            if cards_to_move:
                result.errors.append(f"Failed to move cards: {e}")
            if orphaned:
                result.errors.append(f"Failed to delete orphaned notes: {e}")

    # ---- Phase 4: Return result + pending write ----
    pending = _PendingWrite(
        file_state=fs,
        deck_id_to_write=deck_id_to_write,
        id_assignments=id_assignments,
    )
    return result, pending


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_file(
    file_path: Path,
    only_add_new: bool = False,
) -> SyncResult:
    """Import a single markdown file into Anki."""
    fs = FileState.from_file(file_path)
    anki = AnkiState.fetch()
    converter = MarkdownToHTML()

    # Check for invalid IDs and prompt user
    invalid_ids = FileState.validate_ids(
        [fs],
        valid_deck_ids=set(anki.deck_names_by_id.keys()),
        valid_note_ids=set(anki.notes_by_id.keys()),
    )
    _prompt_invalid_ids(invalid_ids, is_collection=False)

    result, pending = _sync_file(fs, anki, converter, only_add_new=only_add_new)
    _flush_writes([pending])

    return result


def import_collection(
    collection_dir: str,
    only_add_new: bool = False,
) -> ImportSummary:
    """Import all markdown files in a directory back into Anki.

    Single pass:
      1. Parse all files (one read each)
      2. Cross-file validation (duplicate IDs)
      3. Fetch all Anki state (3-4 API calls)
      4. Validate IDs and prompt if needed
      5. Sync each file
      6. Flush all file writes
      7. Detect untracked decks (returned for CLI confirmation)
    """
    collection_path = Path(collection_dir)
    md_files = sorted(collection_path.glob("*.md"))

    # Phase 1: Parse all files
    file_states = [FileState.from_file(f) for f in md_files]

    # Phase 2: Cross-file validation
    global_note_ids: set[int] = set()
    note_id_sources: dict[int, str] = {}
    deck_id_sources: dict[int, str] = {}
    duplicates: list[str] = []
    duplicate_ids: set[int] = set()

    md_deck_ids: set[int] = set()

    for fs in file_states:
        if fs.deck_id is not None:
            md_deck_ids.add(fs.deck_id)
            if fs.deck_id in deck_id_sources:
                duplicates.append(
                    f"Duplicate deck_id {fs.deck_id} found in "
                    f"'{fs.file_path.name}' and "
                    f"'{deck_id_sources[fs.deck_id]}'"
                )
                duplicate_ids.add(fs.deck_id)
            else:
                deck_id_sources[fs.deck_id] = fs.file_path.name

        for note in fs.parsed_notes:
            if note.note_id is not None:
                if note.note_id in note_id_sources:
                    duplicates.append(
                        f"Duplicate note_id {note.note_id} "
                        f"found in '{fs.file_path.name}' and "
                        f"'{note_id_sources[note.note_id]}'"
                    )
                    duplicate_ids.add(note.note_id)
                else:
                    note_id_sources[note.note_id] = fs.file_path.name
                global_note_ids.add(note.note_id)

    if duplicates:
        for dup in duplicates:
            logger.error(dup)
        ids_str = ", ".join(str(i) for i in sorted(duplicate_ids))
        raise ValueError(
            f"Aborting import: {len(duplicate_ids)} duplicate "
            f"ID(s) found across files: {ids_str}. "
            f"Each deck/note ID must appear in exactly one file."
        )

    # Phase 3: Fetch all Anki state
    anki = AnkiState.fetch()
    converter = MarkdownToHTML()

    # Phase 4: Validate IDs and prompt if needed
    invalid_ids = FileState.validate_ids(
        file_states,
        valid_deck_ids=set(anki.deck_names_by_id.keys()),
        valid_note_ids=set(anki.notes_by_id.keys()),
    )
    _prompt_invalid_ids(invalid_ids, is_collection=True)

    # Phase 5: Sync each file
    results: list[SyncResult] = []
    pending_writes: list[_PendingWrite] = []

    for fs in file_states:
        logger.debug(f"Processing {fs.file_path.name}...")
        file_result, pending = _sync_file(
            fs,
            anki,
            converter,
            only_add_new=only_add_new,
            global_note_ids=global_note_ids,
        )
        results.append(file_result)
        pending_writes.append(pending)

        changes = format_changes(
            updated=file_result.updated_count,
            created=file_result.created_count,
            deleted=file_result.deleted_count,
            moved=file_result.moved_count,
            errors=len(file_result.errors),
        )
        if changes != "no changes":
            logger.info(f"  {file_result.deck_name}: {changes}")
        for error in file_result.errors:
            logger.error(f"  {error}")

    # Phase 6: Flush all file writes
    _flush_writes(pending_writes)

    # Phase 7: Detect untracked decks
    untracked_decks: list[UntrackedDeck] = []
    for deck_name, note_ids in anki.note_ids_by_deck_name.items():
        deck_id = anki.deck_ids_by_name.get(deck_name)
        if deck_id is None or deck_id in md_deck_ids:
            continue
        untracked_decks.append(UntrackedDeck(deck_name, deck_id, list(note_ids)))

    return ImportSummary(
        file_results=results,
        untracked_decks=untracked_decks,
    )

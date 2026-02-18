"""Import Markdown files back into Anki.

Architecture:
  AnkiState   – all Anki-side data, fetched once (shared from anki_client)
  FileState   – one markdown file, read once (from models)
  _sync_file  – single engine: classify → update → delete → create
  _flush_writes – deferred file I/O (one write per file, at the end)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki_client import AnkiConnectError, invoke
from ankiops.config import get_collection_dir
from ankiops.key_map import KeyMap
from ankiops.log import clickable_path, format_changes
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import (
    AnkiState,
    Change,
    ChangeType,
    FileState,
    ImportSummary,
    Note,
    SyncResult,
    UntrackedDeck,
)

logger = logging.getLogger(__name__)


@dataclass
class _PendingWrite:
    """Deferred file modification, applied after all Anki mutations."""

    file_state: FileState
    deck_key_to_write: str | None
    key_assignments: list[tuple[Note, str]]


def _flush_writes(writes: list[_PendingWrite]) -> None:
    """Apply all deferred file modifications (one write per file)."""
    for w in writes:
        # If no deck Key to write and no note Key assignments, nothing to do
        if w.deck_key_to_write is None and not w.key_assignments:
            continue

        content = w.file_state.raw_content

        # 1. Insert or replace deck key comment at the top of the file
        if w.deck_key_to_write is not None:
            match_key = re.search(r"^<!--\s*deck_key:.*?-->\n?", content, re.MULTILINE)
            if match_key:
                # Replace existing line
                content = (
                    content[: match_key.start()]
                    + f"<!-- deck_key: {w.deck_key_to_write} -->\n"
                    + content[match_key.end() :]
                )
            else:
                # Prepend new line
                content = f"<!-- deck_key: {w.deck_key_to_write} -->\n" + content

        # 2. Insert keys for new notes

        if w.key_assignments:
            w.file_state.validate_no_duplicate_first_lines(w.key_assignments)

            for note, key_str in w.key_assignments:
                new_key_comment = f"<!-- note_key: {key_str} -->"
                if note.note_key is not None:
                    # Existing key (shouldn't normally happen, but safe)
                    old_key_comment = f"<!-- note_key: {note.note_key} -->"
                    content = content.replace(old_key_comment, new_key_comment, 1)
                else:
                    content = content.replace(
                        note.first_line, new_key_comment + "\n" + note.first_line, 1
                    )

        w.file_state.file_path.write_text(content, encoding="utf-8")


def _build_anki_actions(
    deck_name: str,
    needs_create_deck: bool,
    changes: list[Change],
    anki: AnkiState,
    result: SyncResult,
) -> tuple[list[dict], list[str], list[Change], list[Change], list[int]]:
    """Build the list of actions for AnkiConnect's multi execution.

    Returns:
        (actions, tags, update_changes, create_changes, cards_to_move)
    """
    actions: list[dict] = []
    tags: list[str] = []

    # -- createDeck (must come first) --
    if needs_create_deck:
        actions.append({"action": "createDeck", "params": {"deck": deck_name}})
        tags.append("create_deck")

    # -- changeDeck --
    cards_to_move: list[int] = []
    for change in changes:
        if change.change_type == ChangeType.MOVE:
            # Context must contain 'cards' list
            cards = change.context.get("cards", [])
            cards_to_move.extend(cards)

    if cards_to_move:
        # Deduplicate
        cards_to_move = list(set(cards_to_move))
        actions.append(
            {
                "action": "changeDeck",
                "params": {"cards": cards_to_move, "deck": deck_name},
            }
        )
        tags.append("change_deck")

    # -- updateNoteFields --
    update_changes: list[Change] = []
    for change in changes:
        if change.change_type == ChangeType.UPDATE:
            note_id = change.entity_id
            html_fields = change.context.get("html_fields")
            if note_id and html_fields:
                actions.append(
                    {
                        "action": "updateNoteFields",
                        "params": {"note": {"id": note_id, "fields": html_fields}},
                    }
                )
                update_changes.append(change)
                tags.append("update")

    # -- deleteNotes --
    delete_changes: list[Change] = [
        c for c in changes if c.change_type == ChangeType.DELETE
    ]
    if delete_changes:
        note_ids_to_delete = [
            c.entity_id for c in delete_changes if c.entity_id is not None
        ]
        if note_ids_to_delete:
            actions.append(
                {"action": "deleteNotes", "params": {"notes": note_ids_to_delete}}
            )
            tags.append("delete")

    # -- addNote --
    create_changes: list[Change] = []
    for change in changes:
        if change.change_type == ChangeType.CREATE:
            note = change.context.get("note")
            html_fields = change.context.get("html_fields")
            if note and html_fields:
                # Inject the hidden ID field for safety
                note_key = change.context.get("note_key")
                if note_key:
                    html_fields["AnkiOps Key"] = note_key

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
                create_changes.append(change)
                tags.append("create")

    return actions, tags, update_changes, create_changes, cards_to_move


def _sync_file(
    fs: FileState,
    anki: AnkiState,
    converter: MarkdownToHTML,
    key_map: KeyMap,
    only_add_new: bool = False,
    global_keys: set[str] | None = None,
) -> tuple[SyncResult, _PendingWrite]:
    """Synchronize one markdown file to Anki.

    Phases:
        0. Resolve deck (by deck_id or filename)
        1. Classify notes + convert md → html -> Generate Change objects
        2. Build unified action list
        3. Execute single multi call + route results
        4. Return result + pending file write
    """
    # ---- Phase 0: Resolve deck ----
    deck_key_to_write: str | None = None
    deck_name: str | None = None
    needs_create_deck = False

    # Resolve deck by Key (if present)
    resolved_deck_id = None
    if fs.deck_key:
        resolved_deck_id = key_map.get_deck_id(fs.deck_key)
        if resolved_deck_id:
            deck_name = anki.deck_names_by_id.get(resolved_deck_id)
            if deck_name:
                logger.debug(f"Resolved deck matching Key {fs.deck_key}: {deck_name}")
            else:
                # Deck ID found in map, but not in Anki (deleted).
                # Treat as unresolved so we fall back to filename matching.
                logger.warning(
                    f"Deck Key {fs.deck_key} maps to ID {resolved_deck_id} "
                    "which does not exist in Anki. Treating as new/unresolved."
                )
                resolved_deck_id = None

    # If not resolved by Key, try by filename
    if not resolved_deck_id:
        deck_name = fs.file_path.stem.replace("__", "::")
        resolved_deck_id = anki.deck_ids_by_name.get(deck_name)

        if resolved_deck_id:
            # Found by name. Logic to assign Key below.
            pass
        else:
            needs_create_deck = True

    # Ensure deck has Key assignment
    if resolved_deck_id:
        # Check mapping
        current_key = key_map.get_deck_key(resolved_deck_id)

        if not current_key:
            if fs.deck_key:
                # Import existing Key from file
                key_map.set_deck(fs.deck_key, resolved_deck_id)
            else:
                # Generate new Key
                new_key = KeyMap.generate_key()
                key_map.set_deck(new_key, resolved_deck_id)
                deck_key_to_write = new_key

            # File has wrong Key or no Key, strict sync enforces mapped Key
            deck_key_to_write = current_key

    elif needs_create_deck:
        # Creating a new deck, need to assign a Key for it
        if fs.deck_key:
            deck_key_to_write = fs.deck_key
        else:
            deck_key_to_write = KeyMap.generate_key()

    result = SyncResult(
        deck_name=deck_name,
        file_path=fs.file_path,
    )

    # ---- Phase 1: Classify + convert ----
    changes: list[Change] = []

    # Process parsed notes (Creates, Updates, Skips, Moves)
    for parsed_note in fs.parsed_notes:
        if errs := parsed_note.validate():
            for e in errs:
                result.errors.append(f"Note {parsed_note.identifier}: {e}")
            continue
        try:
            html_fields = parsed_note.to_html(converter)
        except Exception as e:
            result.errors.append(f"Note {parsed_note.identifier}: {e}")
            continue

        if parsed_note.note_key:
            # Note has a Key — look up the Anki note_id from mapping
            note_id = key_map.get_note_id(parsed_note.note_key)
            found_ids = None

            # --- RECOVERY LOGIC START ---
            if note_id is None:
                # Key exists in markdown but not in ID Map.
                # Check if it exists in Anki via the "AnkiOps Key" field (crash recovery).
                try:
                    # Query for the specific ID (exact match)
                    found_ids = invoke(
                        "findNotes", query=f'"AnkiOps Key:{parsed_note.note_key}"'
                    )
                    if found_ids:
                        note_id = found_ids[0]
                        logger.info(
                            f"Recovered link for key {parsed_note.note_key} -> note_id {note_id}"
                        )
                        # Immediately update the map so subsequent lookups (and moves/updates) work
                        key_map.set_note(parsed_note.note_key, note_id)
                except AnkiConnectError as e:
                    logger.warning(
                        f"Recovery lookup failed for {parsed_note.note_key}: {e}"
                    )
            # --- RECOVERY LOGIC END ---

            if note_id and only_add_new:
                changes.append(Change(ChangeType.SKIP, note_id, parsed_note.identifier))
                continue

            if note_id:
                anki_note = anki.notes_by_id.get(note_id)
                # If recovery just happened, anki_note won't be in the potentially stale AnkiState.
                # We need to fetch it or create a placeholder if we want to support updates/moves immediately.
                # Re-fetching everything is expensive.
                # Strategy: If it was recovered, we assume it exists. If it's missing from AnkiState,
                # we technically can't compare fields easily without a partial fetch.
                # However, for safety/simplicity in this edge case:
                # If we recovered it, we know the ID is valid.
                if not anki_note and found_ids:  # type: ignore (found_ids might be referenced)
                    # We can try to fetch just this note's info
                    try:
                        info_list = invoke("notesInfo", notes=[note_id])
                        if info_list and info_list[0]:
                            from ankiops.models import (
                                AnkiNote,  # Deferred import to avoid circularity if any
                            )

                            anki_note = AnkiNote.from_raw(info_list[0])
                            # cache it back
                            anki.notes_by_id[note_id] = anki_note
                    except Exception as e:
                        logger.warning(f"Could not fetch recovered note details: {e}")

            else:
                anki_note = None

            if not anki_note:
                # Key exists but no valid Anki note — treat as new (create)
                changes.append(
                    Change(
                        ChangeType.CREATE,
                        None,
                        parsed_note.identifier,
                        context={
                            "note": parsed_note,
                            "html_fields": html_fields,
                            "note_key": parsed_note.note_key,
                        },
                    )
                )
            else:
                # Note exists in Anki

                # Check for deck movement
                cards_to_move = []
                for cid in anki_note.card_ids:
                    card = anki.cards_by_id.get(cid)
                    if card and card.get("deckName") != deck_name:
                        cards_to_move.append(cid)

                if cards_to_move:
                    changes.append(
                        Change(
                            ChangeType.MOVE,
                            note_id,
                            parsed_note.identifier,
                            context={"cards": cards_to_move},
                        )
                    )

                # Check for type mismatch
                if anki_note.note_type != parsed_note.note_type:
                    result.errors.append(
                        f"Note type mismatch for note {parsed_note.identifier}: "
                        f"Markdown specifies '{parsed_note.note_type}' "
                        f"but Anki has '{anki_note.note_type}'."
                    )
                    continue

                # --- FIX: Ensure AnkiOps Key is set ---
                # Even if other content matches, we must ensure the Key field matches.
                # If it's missing or wrong in Anki, we treat it as an update.
                current_anki_key_val = anki_note.fields.get("AnkiOps Key", "")
                if current_anki_key_val != parsed_note.note_key:
                    html_fields["AnkiOps Key"] = parsed_note.note_key
                    # Merge logic: if content is same but Key diff, we force update

                # Check for content update
                if parsed_note.html_fields_match(html_fields, anki_note):
                    changes.append(
                        Change(ChangeType.SKIP, note_id, parsed_note.identifier)
                    )
                else:
                    changes.append(
                        Change(
                            ChangeType.UPDATE,
                            note_id,
                            parsed_note.identifier,
                            context={"html_fields": html_fields},
                        )
                    )

        else:
            # New note (no Key yet)
            new_key = KeyMap.generate_key()
            changes.append(
                Change(
                    ChangeType.CREATE,
                    None,
                    parsed_note.identifier,
                    context={
                        "note": parsed_note,
                        "html_fields": html_fields,
                        "note_key": new_key,  # Pre-generated for injection
                    },
                )
            )

    # Identify orphans (Deletes)
    # Build the set of Anki note_ids that are referenced by this file's Keys
    md_anki_ids = set()
    for key_str in fs.note_keys:
        nid = key_map.get_note_id(key_str)
        if nid is not None:
            md_anki_ids.add(nid)

    anki_deck_note_ids = anki.note_ids_by_deck_name.get(deck_name, set()).copy()
    orphaned = anki_deck_note_ids - md_anki_ids

    # Exclude notes referenced by other files in global context
    if global_keys:
        global_anki_ids = set()
        for u in global_keys:
            nid = key_map.get_note_id(u)
            if nid is not None:
                global_anki_ids.add(nid)
        orphaned -= global_anki_ids

    for nid in orphaned:
        orphan_key = key_map.get_note_key(nid)
        repr_str = f"note_key: {orphan_key}" if orphan_key else f"note_id: {nid}"
        changes.append(Change(ChangeType.DELETE, nid, repr_str))

    # Store changes in result
    result.changes = changes

    # ---- Phase 2: Build unified action list ----
    if deck_name is None:
        raise ValueError("Deck name could not be resolved.")

    actions, tags, update_changes, create_changes, cards_to_move = _build_anki_actions(
        deck_name, needs_create_deck, changes, anki, result
    )

    # ---- Phase 3: Execute single multi call + route results ----
    key_assignments: list[tuple[Note, str]] = []

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

                    if deck_key_to_write:
                        key_map.set_deck(deck_key_to_write, new_deck_id)

                    logger.info(f"Created new deck '{deck_name}' (id: {new_deck_id})")

                elif tag == "change_deck":
                    if res is not None:
                        result.errors.append(f"Failed to move cards: {res}")
                    else:
                        logger.debug(f"Moved cards to '{deck_name}'")

                elif tag == "update":
                    change = update_changes[update_idx]
                    update_idx += 1
                    if res is not None:
                        result.errors.append(
                            f"Note {change.entity_id} ({change.entity_repr}): {res}"
                        )

                elif tag == "delete":
                    if res is not None:
                        result.errors.append(f"Failed to delete orphaned notes: {res}")

                elif tag == "create":
                    change = create_changes[create_idx]
                    create_idx += 1
                    note = change.context.get("note")
                    if (
                        res
                        and isinstance(res, int)
                        and note
                        and (key_str := change.context.get("note_key"))
                    ):
                        # Assign Key: use pre-generated key
                        key_map.set_note(key_str, res)
                        key_assignments.append((note, key_str))
                        change.entity_id = res

                    else:
                        result.errors.append(f"Note new ({change.entity_repr}): {res}")

        except AnkiConnectError as e:
            # The entire multi call failed — attribute errors to all actions
            for change in update_changes:
                result.errors.append(
                    f"Note {change.entity_id} ({change.entity_repr}): {e}"
                )
            for change in create_changes:
                result.errors.append(f"Note new ({change.entity_repr}): {e}")
            if cards_to_move:
                result.errors.append(f"Failed to move cards: {e}")
            if orphaned:
                result.errors.append(f"Failed to delete orphaned notes: {e}")

    # ---- Phase 4: Return result + pending write ----
    pending = _PendingWrite(
        file_state=fs,
        deck_key_to_write=deck_key_to_write,
        key_assignments=key_assignments,
    )

    return result, pending


def import_file(
    file_path: Path,
    only_add_new: bool = False,
) -> SyncResult:
    """Import a single markdown file into Anki."""
    fs = FileState.from_file(file_path)
    anki = AnkiState.fetch()
    converter = MarkdownToHTML()
    collection_dir = get_collection_dir()
    key_map = KeyMap.load(collection_dir)

    result, pending = _sync_file(
        fs, anki, converter, key_map, only_add_new=only_add_new
    )
    _flush_writes([pending])
    key_map.save(collection_dir)

    return result


def import_collection(
    collection_dir: str,
    only_add_new: bool = False,
) -> ImportSummary:
    """Import all markdown files in a directory back into Anki.

    Single pass:
      1. Parse all files (one read each)
      2. Cross-file validation (duplicate Keys)
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
    global_keys: set[str] = set()
    key_sources: dict[str, str] = {}
    deck_id_sources: dict[int, str] = {}
    duplicates: list[str] = []
    duplicate_ids: set[str | int] = set()

    md_deck_ids: set[int] = set()

    for fs in file_states:
        # Check for duplicate deck Keys
        if fs.deck_key:
            if fs.deck_key in key_sources:
                duplicates.append(
                    f"Duplicate deck_key {fs.deck_key} found in "
                    f"'{fs.file_path.name}' and "
                    f"'{key_sources[fs.deck_key]}'"
                )
            else:
                key_sources[fs.deck_key] = fs.file_path.name

        for note in fs.parsed_notes:
            if note.note_key is not None:
                if note.note_key in key_sources:
                    duplicates.append(
                        f"Duplicate note_key {note.note_key} "
                        f"found in '{fs.file_path.name}' and "
                        f"'{key_sources[note.note_key]}'"
                    )
                    duplicate_ids.add(note.note_key)
                else:
                    key_sources[note.note_key] = fs.file_path.name
                global_keys.add(note.note_key)

    if duplicates:
        for dup in duplicates:
            logger.error(dup)
        ids_str = ", ".join(str(i) for i in sorted(duplicate_ids, key=str))
        raise ValueError(
            f"Aborting import: {len(duplicate_ids)} duplicate "
            f"Key(s) found across files: {ids_str}. "
            f"Each deck/Key must appear in exactly one file."
        )

    # Phase 3: Fetch all Anki state
    anki = AnkiState.fetch()
    converter = MarkdownToHTML()
    key_map = KeyMap.load(collection_path)

    # Phase 5: Sync each file
    results: list[SyncResult] = []
    pending_writes: list[_PendingWrite] = []

    for fs in file_states:
        logger.debug(f"Processing {fs.file_path.name}...")
        file_result, pending = _sync_file(
            fs,
            anki,
            converter,
            key_map,
            only_add_new=only_add_new,
            global_keys=global_keys,
        )
        if file_result.deck_name in anki.deck_ids_by_name:
            md_deck_ids.add(anki.deck_ids_by_name[file_result.deck_name])

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
    key_map.save(collection_path)

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

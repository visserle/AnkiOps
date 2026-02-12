"""Import Markdown files back into Anki."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from deckops.anki_client import invoke
from deckops.config import NOTE_SEPARATOR, SUPPORTED_NOTE_TYPES
from deckops.markdown_converter import MarkdownToHTML
from deckops.markdown_helpers import (
    ParsedNote,
    extract_deck_id,
    parse_note_block,
    validate_note,
)

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of importing a single note."""

    note_id: int | None
    success: bool
    action: str  # "updated", "created", "skipped", "error"
    message: str = ""


@dataclass
class FileImportResult:
    """Result of importing an entire file."""

    file_path: Path
    total_notes: int
    updated: int
    created: int
    deleted: int
    skipped: int
    errors: list[str] = field(default_factory=list)


def parse_deck_id(file_path: Path) -> int | None:
    """Extract deck_id from the first line of a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    deck_id, _ = extract_deck_id(content)
    return deck_id


def write_deck_id_to_file(file_path: Path, deck_id: int) -> None:
    """Write or replace the deck_id on the first line of a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    _, remaining = extract_deck_id(content)
    new_first_line = "<!-- deck_id: {} -->".format(deck_id) + "\n"
    file_path.write_text(new_first_line + remaining, encoding="utf-8")


def parse_markdown_file(file_path: Path) -> list[ParsedNote]:
    """Parse a markdown file into a list of ParsedNote objects."""
    content = file_path.read_text(encoding="utf-8")
    _, content = extract_deck_id(content)
    blocks = content.split(NOTE_SEPARATOR)

    parsed_notes = []
    line_number = 1

    for block in blocks:
        if block.strip():
            parsed_notes.append(parse_note_block(block, line_number))
        line_number += block.count("\n") + NOTE_SEPARATOR.count("\n")

    return parsed_notes


def convert_fields_to_html(
    fields: dict[str, str],
    converter: MarkdownToHTML,
) -> dict[str, str]:
    """Convert all field values from markdown to HTML."""
    return {name: converter.convert(content) for name, content in fields.items()}


def update_note(note_id: int, fields: dict[str, str]) -> None:
    """Update a note's fields in Anki."""
    invoke("updateNoteFields", note={"id": note_id, "fields": fields})


def create_note(
    deck_name: str,
    model_name: str,
    fields: dict[str, str],
) -> int:
    """Create a new note in Anki. Returns the new note ID."""
    result = invoke(
        "addNote",
        note={
            "deckName": deck_name,
            "modelName": model_name,
            "fields": fields,
            "options": {"allowDuplicate": False},
        },
    )
    return result


def _batch_write_ids_to_file(
    file_path: Path,
    id_assignments: list[tuple[ParsedNote, int]],
) -> None:
    """Write note_ids back into the markdown file in a single pass.

    Handles two cases:
    - New notes (no old ID): prepend the ID comment before the first field line
    - Stale notes (had old ID deleted from Anki): replace the old ID with new one
    """
    if not id_assignments:
        return

    # VALIDATION: Check for duplicate first lines among new notes
    # This would cause only the first note to get an ID, orphaning the others
    new_notes_first_lines: dict[str, list[int]] = {}
    for parsed_note, id_value in id_assignments:
        if parsed_note.note_id is None:  # New note without existing ID
            first_line = parsed_note.raw_content.strip().split("\n")[0]
            if first_line not in new_notes_first_lines:
                new_notes_first_lines[first_line] = []
            new_notes_first_lines[first_line].append(parsed_note.line_number)

    # Check for duplicates
    duplicates = {
        line: nums for line, nums in new_notes_first_lines.items() if len(nums) > 1
    }
    if duplicates:
        error_msg = f"ERROR: Duplicate first lines detected in {file_path.name}:\n"
        for first_line, line_numbers in duplicates.items():
            error_msg += f"  '{first_line[:60]}...' appears at lines: {line_numbers}\n"
        error_msg += (
            "Cannot safely assign IDs. Please ensure each note has a unique first line."
        )
        raise ValueError(error_msg)

    content = file_path.read_text(encoding="utf-8")
    for parsed_note, id_value in id_assignments:
        new_id_comment = f"<!-- note_id: {id_value} -->"

        if parsed_note.note_id is not None:
            # Stale note: replace old ID comment with new one
            old_id_comment = f"<!-- note_id: {parsed_note.note_id} -->"
            content = content.replace(old_id_comment, new_id_comment, 1)
        else:
            # New note: prepend ID comment before first content line
            first_line = parsed_note.raw_content.strip().split("\n")[0]
            if first_line in content:
                content = content.replace(
                    first_line, new_id_comment + "\n" + first_line, 1
                )
    file_path.write_text(content, encoding="utf-8")


def _import_existing_notes(
    existing_notes: list[tuple[ParsedNote, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedNote, dict[str, str]]]:
    """Phase 1: Batch update existing notes (both QA and Cloze).

    Returns stale entries (note_id no longer in Anki) for re-creation.
    Also moves cards to the correct deck if they were moved between files.
    """
    if not existing_notes:
        return []

    existing_note_ids = [parsed_note.note_id for parsed_note, _ in existing_notes]

    try:
        notes_info = invoke("notesInfo", notes=existing_note_ids)
    except Exception as e:
        for parsed_note, _ in existing_notes:
            result.errors.append(
                f"Note {parsed_note.note_id} (line {parsed_note.line_number}): {e}"
            )
        return []

    # Safety check: Ensure all notes have a correct DeckOps template
    for note in notes_info:
        model = note.get("modelName") if note else None
        if model and model not in SUPPORTED_NOTE_TYPES:
            raise ValueError(
                f"Safety check failed: Note {note['noteId']} has template "
                f"'{note['modelName']}' but expected a DeckOps template. "
                f"DeckOps will never modify notes with non-DeckOps templates."
            )

    note_fields: dict[int, dict[str, str]] = {}
    note_card_ids: dict[int, list[int]] = {}
    for note in notes_info:
        if note:
            note_fields[note["noteId"]] = {
                name: info["value"] for name, info in note["fields"].items()
            }
            note_card_ids[note["noteId"]] = note.get("cards", [])

    # Move cards that are in the wrong deck
    all_card_ids = [cid for cids in note_card_ids.values() for cid in cids]
    if all_card_ids:
        try:
            cards_info = invoke("cardsInfo", cards=all_card_ids)
            cards_to_move = [
                c["cardId"]
                for c in cards_info
                if c and c.get("deckName") and c["deckName"] != deck_name
            ]
            if cards_to_move:
                invoke("changeDeck", cards=cards_to_move, deck=deck_name)
                card_to_note_map = {
                    cid: nid for nid, cids in note_card_ids.items() for cid in cids
                }
                card_to_deck_map = {c["cardId"]: c["deckName"] for c in cards_info if c}
                moved_notes = set()
                for cid in cards_to_move:
                    nid = card_to_note_map.get(cid)
                    if nid and nid not in moved_notes:
                        moved_notes.add(nid)
                        logger.info(
                            f"  Moved note {nid} "
                            f"from '{card_to_deck_map.get(cid, '?')}' to '{deck_name}'"
                        )
        except Exception as e:
            for parsed_note, _ in existing_notes:
                result.errors.append(
                    f"Note {parsed_note.note_id}: failed to move deck: {e}"
                )

    updates: list[dict] = []
    notes_to_update: list[ParsedNote] = []
    stale: list[tuple[ParsedNote, dict[str, str]]] = []

    for parsed_note, html_fields in existing_notes:
        assert parsed_note.note_id is not None
        current = note_fields.get(parsed_note.note_id, {})
        if not current:
            logger.info(
                f"  Note {parsed_note.note_id} "
                f"(line {parsed_note.line_number}) "
                f"no longer in Anki, will re-create"
            )
            stale.append((parsed_note, html_fields))
            continue

        if all(current.get(name) == value for name, value in html_fields.items()):
            result.skipped += 1
            continue

        updates.append(
            {
                "action": "updateNoteFields",
                "params": {
                    "note": {
                        "id": parsed_note.note_id,
                        "fields": html_fields,
                    }
                },
            }
        )
        notes_to_update.append(parsed_note)

    if updates:
        try:
            multi_results = invoke("multi", actions=updates)
            for i, res in enumerate(multi_results):
                if res is None:
                    result.updated += 1
                else:
                    result.errors.append(
                        f"Note {notes_to_update[i].note_id} "
                        f"(line {notes_to_update[i].line_number}): "
                        f"{res}"
                    )
        except Exception as e:
            for parsed_note in notes_to_update:
                result.errors.append(
                    f"Note {parsed_note.note_id} (line {parsed_note.line_number}): {e}"
                )

    return stale


def _create_new_notes(
    new_notes: list[tuple[ParsedNote, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedNote, int]]:
    """Phase 3: Batch create new notes (both QA and Cloze).

    Returns (parsed_note, note_id) pairs for ID writeback.
    """
    if not new_notes:
        return []

    create_actions = [
        {
            "action": "addNote",
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": parsed_note.note_type,
                    "fields": html_fields,
                    "options": {"allowDuplicate": False},
                }
            },
        }
        for parsed_note, html_fields in new_notes
    ]

    try:
        create_results = invoke("multi", actions=create_actions)
    except Exception as e:
        for parsed_note, _ in new_notes:
            result.errors.append(f"Note new (line {parsed_note.line_number}): {e}")
        return []

    note_id_assignments: list[tuple[ParsedNote, int]] = []
    for i, note_id in enumerate(create_results):
        parsed_note, _ = new_notes[i]
        if note_id and isinstance(note_id, int):
            note_id_assignments.append((parsed_note, note_id))
            result.created += 1
        else:
            result.errors.append(
                f"Note new (line {parsed_note.line_number}): {note_id}"
            )

    return note_id_assignments


def _delete_orphaned_notes(
    parsed_notes: list[ParsedNote],
    deck_name: str,
    result: FileImportResult,
    global_note_ids: set[int] | None = None,
) -> None:
    """Delete notes from Anki that are no longer in the markdown file.
    (cleanup within tracked files)

    Must run BEFORE creating new notes, so newly created notes
    can't be mistaken for orphans.

    When global_note_ids is provided (collection mode),
    notes claimed by other files are excluded from deletion — those files
    will move the notes to the correct deck when they are processed.
    """
    md_note_ids = {n.note_id for n in parsed_notes if n.note_id is not None}

    for note_type in SUPPORTED_NOTE_TYPES:
        query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:{note_type}'
        anki_card_ids = invoke("findCards", query=query)

        if not anki_card_ids:
            continue

        cards_info = invoke("cardsInfo", cards=anki_card_ids)
        anki_note_ids = {c["note"] for c in cards_info}
        orphaned_notes = anki_note_ids - md_note_ids
        if global_note_ids:
            orphaned_notes -= global_note_ids

        if orphaned_notes:
            # Build note_id -> card_ids mapping for clear logging
            note_to_cards: dict[int, list[int]] = {}
            for c in cards_info:
                if c["note"] in orphaned_notes:
                    note_to_cards.setdefault(c["note"], []).append(c["cardId"])

            invoke("deleteNotes", notes=list(orphaned_notes))
            result.deleted += len(orphaned_notes)
            for nid, cids in note_to_cards.items():
                cid_str = ", ".join(str(c) for c in cids)
                logger.info(
                    f"  Deleted {note_type} note {nid}"
                    f"{f' (cards {cid_str})' if cid_str else ''}"
                    f" from Anki"
                )


def import_file(
    file_path: Path,
    deck_name: str | None = None,
    only_add_new: bool = False,
    deck_names_and_ids: dict[str, int] | None = None,
    global_note_ids: set[int] | None = None,
) -> FileImportResult:
    """Import all notes from a markdown file into Anki.

    When global_note_ids is provided, orphan deletion will not remove
    notes that are tracked by other files in the collection — they
    will be moved to the correct deck instead.
    """
    # Auto-detect deck name
    if deck_name is None:
        if deck_names_and_ids is None:
            deck_names_and_ids = invoke("deckNamesAndIds")
        assert deck_names_and_ids is not None
        id_to_name = {v: k for k, v in deck_names_and_ids.items()}

        file_deck_id = parse_deck_id(file_path)
        if file_deck_id and file_deck_id in id_to_name:
            deck_name = id_to_name[file_deck_id]
            logger.debug(f"Resolved deck by ID {file_deck_id}: {deck_name}")
        else:
            deck_name = file_path.stem.replace("__", "::")

            if deck_name not in deck_names_and_ids:
                new_deck_id = invoke("createDeck", deck=deck_name)
                write_deck_id_to_file(file_path, new_deck_id)
                deck_names_and_ids[deck_name] = new_deck_id
                logger.info(f"Created new deck '{deck_name}' (id: {new_deck_id})")
            elif not file_deck_id:
                existing_deck_id = deck_names_and_ids[deck_name]
                write_deck_id_to_file(file_path, existing_deck_id)
                logger.debug(f"Wrote deck_id {existing_deck_id} to {file_path.name}")

    parsed_notes = parse_markdown_file(file_path)
    converter = MarkdownToHTML()

    result = FileImportResult(
        file_path=file_path,
        total_notes=len(parsed_notes),
        updated=0,
        created=0,
        deleted=0,
        skipped=0,
    )

    # Phase 0: Classify notes and convert fields
    existing: list[tuple[ParsedNote, dict[str, str]]] = []
    new: list[tuple[ParsedNote, dict[str, str]]] = []

    for parsed_note in parsed_notes:
        validation_errors = validate_note(parsed_note)
        if validation_errors:
            for err in validation_errors:
                result.errors.append(
                    f"Note {parsed_note.note_id or 'new'} "
                    f"(line {parsed_note.line_number}): {err}"
                )
            continue

        try:
            html_fields = convert_fields_to_html(parsed_note.fields, converter)
        except Exception as e:
            result.errors.append(
                f"Note {parsed_note.note_id or 'new'} "
                f"(line {parsed_note.line_number}): {e}"
            )
            continue

        if parsed_note.note_id:
            if only_add_new:
                result.skipped += 1
                continue
            existing.append((parsed_note, html_fields))
        elif deck_name:
            new.append((parsed_note, html_fields))
        else:
            result.errors.append(
                f"Note new (line {parsed_note.line_number}): No deck_name"
            )

    # Phase 1: Update existing notes (returns stale entries)
    stale = _import_existing_notes(existing, deck_name, result)
    new.extend(stale)

    # Phase 2: Delete orphaned notes
    # (BEFORE creation so new notes aren't mistaken for orphans)
    if deck_name:
        _delete_orphaned_notes(
            parsed_notes,
            deck_name,
            result,
            global_note_ids=global_note_ids,
        )

    # Phase 3: Create new notes (includes stale re-creations)
    id_assignments = _create_new_notes(new, deck_name, result)

    # Phase 4: Write IDs back to file
    _batch_write_ids_to_file(file_path, id_assignments)

    return result


def delete_untracked_notes(collection_dir: str) -> int:
    """Delete DeckOpsQA/DeckOpsCloze notes from Anki decks that have no markdown file.
    (cleanup of untracked decks, deck-level)

    Only affects notes with managed note types (DeckOpsQA, DeckOpsCloze).
    Other note types in the same deck are left untouched.
    Only targets decks whose deck_id does not match any file's deck_id
    in the collection directory.
    """
    collection_path = Path(collection_dir)

    md_deck_ids: set[int] = set()
    for md_file in collection_path.glob("*.md"):
        deck_id = parse_deck_id(md_file)
        if deck_id is not None:
            md_deck_ids.add(deck_id)

    deck_names_and_ids = invoke("deckNamesAndIds")

    # Single query to find all cards with supported note types
    query = " OR ".join(f"note:{nt}" for nt in SUPPORTED_NOTE_TYPES)
    all_card_ids = invoke("findCards", query=query)

    orphaned: list[tuple[str, int, list[int]]] = []
    if all_card_ids:
        cards_info = invoke("cardsInfo", cards=all_card_ids)
        # Group note IDs by deck name
        deck_notes: dict[str, set[int]] = {}
        for card in cards_info:
            deck_notes.setdefault(card["deckName"], set()).add(card["note"])

        for deck_name, note_ids in deck_notes.items():
            deck_id = deck_names_and_ids.get(deck_name)
            if deck_id is None or deck_id in md_deck_ids:
                continue
            orphaned.append((deck_name, deck_id, list(note_ids)))

    if not orphaned:
        return 0

    logger.warning(
        "The following Anki decks with DeckOpsQA/DeckOpsCloze notes inside have no matching"
        " markdown file:"
    )
    for deck_name, deck_id, note_ids in orphaned:
        logger.warning(
            f"  - '{deck_name}' (deck_id: {deck_id}, {len(note_ids)} managed notes)"
        )

    answer = (
        input(
            "Delete these managed notes from Anki (only DeckOpsQA/DeckOpsCloze "
            "notes will be removed)? [y/N] "
        )
        .strip()
        .lower()
    )
    if answer != "y":
        logger.info("Skipped untracked note deletion.")
        return 0

    total_deleted = 0
    for deck_name, deck_id, note_ids in orphaned:
        invoke("deleteNotes", notes=note_ids)
        total_deleted += len(note_ids)
        logger.info(
            f"  Deleted {len(note_ids)} managed notes from '{deck_name}'"
            f" (deck_id: {deck_id})"
        )

    return total_deleted


def import_collection(
    collection_dir: str,
    only_add_new: bool = False,
) -> list[FileImportResult]:
    """Import all markdown files in a directory back into Anki."""
    collection_path = Path(collection_dir)
    deck_names_and_ids = invoke("deckNamesAndIds")
    results = []

    # Pre-parse all files to collect globally tracked IDs.
    # This prevents orphan deletion from removing notes that were
    # moved between files — the claiming file will move them instead.
    # Also detect duplicate IDs across files and abort if found.
    md_files = sorted(collection_path.glob("*.md"))
    global_note_ids: set[int] = set()
    note_id_sources: dict[int, str] = {}  # note_id -> filename
    deck_id_sources: dict[int, str] = {}  # deck_id -> filename
    duplicates: list[str] = []
    duplicate_ids: set[int] = set()

    # Check for duplicate deck_ids across files
    for md_file in md_files:
        deck_id = parse_deck_id(md_file)
        if deck_id is not None:
            if deck_id in deck_id_sources:
                duplicates.append(
                    f"Duplicate deck_id {deck_id} found in "
                    f"'{md_file.name}' and '{deck_id_sources[deck_id]}'"
                )
                duplicate_ids.add(deck_id)
            else:
                deck_id_sources[deck_id] = md_file.name

    for md_file in md_files:
        for parsed_note in parse_markdown_file(md_file):
            if parsed_note.note_id is not None:
                if parsed_note.note_id in note_id_sources:
                    duplicates.append(
                        f"Duplicate note_id {parsed_note.note_id} "
                        f"found in '{md_file.name}' and "
                        f"'{note_id_sources[parsed_note.note_id]}'"
                    )
                    duplicate_ids.add(parsed_note.note_id)
                else:
                    note_id_sources[parsed_note.note_id] = md_file.name
                global_note_ids.add(parsed_note.note_id)
    if duplicates:
        for dup in duplicates:
            logger.error(dup)
        ids_str = ", ".join(str(i) for i in sorted(duplicate_ids))
        raise ValueError(
            f"Aborting import: {len(duplicate_ids)} duplicate ID(s) found across files:"
            f" {ids_str}. Each deck/note ID must appear in exactly one file."
        )

    for md_file in md_files:
        logger.info(f"Processing {md_file.name}...")
        result = import_file(
            md_file,
            only_add_new=only_add_new,
            deck_names_and_ids=deck_names_and_ids,
            global_note_ids=global_note_ids,
        )
        results.append(result)

        logger.info(
            f"  Updated: {result.updated}, Created: {result.created}, "
            f"Deleted: {result.deleted}, Skipped: {result.skipped}, "
            f"Errors: {len(result.errors)}"
        )

        for error in result.errors:
            logger.error(f"  {error}")

    return results

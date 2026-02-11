"""Import Markdown files back into Anki."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from deckops.anki_client import extract_deck_id, invoke
from deckops.config import CARD_SEPARATOR, NOTE_TYPES, SUPPORTED_NOTE_TYPES
from deckops.markdown_converter import MarkdownToHTML
from deckops.markdown_helpers import ParsedCard, parse_card_block, validate_card

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of importing a single card."""

    card_id: int | None
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


def parse_markdown_file(file_path: Path) -> list[ParsedCard]:
    """Parse a markdown file into a list of ParsedCard objects."""
    content = file_path.read_text(encoding="utf-8")
    _, content = extract_deck_id(content)
    blocks = content.split(CARD_SEPARATOR)

    cards = []
    line_number = 1

    for block in blocks:
        if block.strip():
            cards.append(parse_card_block(block, line_number))
        line_number += block.count("\n") + CARD_SEPARATOR.count("\n")

    return cards


def convert_fields_to_html(
    fields: dict[str, str],
    converter: MarkdownToHTML,
) -> dict[str, str]:
    """Convert all field values from markdown to HTML."""
    return {name: converter.convert(content) for name, content in fields.items()}


def get_note_id_for_card(card_id: int) -> int:
    """Get the note ID for a given card ID."""
    cards_info = invoke("cardsInfo", cards=[card_id])
    if not cards_info:
        raise ValueError(f"Card {card_id} not found in Anki")
    return cards_info[0]["note"]


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


def get_card_id_for_note(note_id: int) -> int | None:
    """Get the first card ID for a note."""
    notes_info = invoke("notesInfo", notes=[note_id])
    if notes_info and notes_info[0].get("cards"):
        return notes_info[0]["cards"][0]
    return None


def _batch_write_ids_to_file(
    file_path: Path,
    id_assignments: list[tuple[ParsedCard, int]],
) -> None:
    """Write card_ids or note_ids back into the markdown file in a single pass.

    Handles two cases:
    - New cards (no old ID): prepend the ID comment before the first field line
    - Stale cards (had old ID deleted from Anki): replace the old ID with new one
    """
    if not id_assignments:
        return

    # VALIDATION: Check for duplicate first lines among new cards
    # This would cause only the first card to get an ID, orphaning the others
    new_cards_first_lines: dict[str, list[int]] = {}
    for card, id_value in id_assignments:
        old_id = card.card_id if card.note_type == "DeckOpsQA" else card.note_id
        if old_id is None:  # New card without existing ID
            first_line = card.raw_content.strip().split("\n")[0]
            if first_line not in new_cards_first_lines:
                new_cards_first_lines[first_line] = []
            new_cards_first_lines[first_line].append(card.line_number)

    # Check for duplicates
    duplicates = {
        line: nums for line, nums in new_cards_first_lines.items() if len(nums) > 1
    }
    if duplicates:
        error_msg = f"ERROR: Duplicate first lines detected in {file_path.name}:\n"
        for first_line, line_numbers in duplicates.items():
            error_msg += f"  '{first_line[:60]}...' appears at lines: {line_numbers}\n"
        error_msg += (
            "Cannot safely assign IDs. Please ensure each card has a unique first line."
        )
        raise ValueError(error_msg)

    content = file_path.read_text(encoding="utf-8")
    for card, id_value in id_assignments:
        id_type = NOTE_TYPES[card.note_type]["id_type"]
        new_id_comment = f"<!-- {id_type}: {id_value} -->"

        old_id = card.card_id if card.note_type == "DeckOpsQA" else card.note_id
        if old_id is not None:
            # Stale card: replace old ID comment with new one
            # Stale means here the card/note was deleted from Anki and re-created with a
            # new ID, so we need to update the markdown file to reflect the new ID
            old_id_comment = f"<!-- {id_type}: {old_id} -->"
            content = content.replace(old_id_comment, new_id_comment, 1)
        else:
            # New card: prepend ID comment before first content line
            first_line = card.raw_content.strip().split("\n")[0]
            if first_line in content:
                content = content.replace(
                    first_line, new_id_comment + "\n" + first_line, 1
                )
    file_path.write_text(content, encoding="utf-8")


def _import_existing_qa_cards(
    existing_cards: list[tuple[ParsedCard, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedCard, dict[str, str]]]:
    """Phase 1: Batch update existing DeckOpsQA cards.

    Returns stale entries (card_id no longer in Anki) for re-creation.
    Also moves cards to the correct deck if they were moved between files.
    """
    if not existing_cards:
        return []

    existing_card_ids = [card.card_id for card, _ in existing_cards]

    try:
        cards_info = invoke("cardsInfo", cards=existing_card_ids)
    except Exception as e:
        for card, _ in existing_cards:
            result.errors.append(f"Card {card.card_id} (line {card.line_number}): {e}")
        return []

    # Safety check: Ensure all cards have the correct DeckOps template
    for info in cards_info:
        if info and info.get("modelName") and info["modelName"] != "DeckOpsQA":
            raise ValueError(
                f"Safety check failed: Card {info['cardId']} has template "
                f"'{info['modelName']}' but expected 'DeckOpsQA'. "
                f"DeckOps will never modify cards with non-DeckOps templates."
            )

    card_to_note: dict[int, int] = {}
    card_to_deck: dict[int, str] = {}
    for info in cards_info:
        if info and info.get("note"):
            card_to_note[info["cardId"]] = info["note"]
            card_to_deck[info["cardId"]] = info.get("deckName", "")

    # Move cards that are in the wrong deck
    cards_to_move = [
        cid
        for cid, current_deck in card_to_deck.items()
        if current_deck and current_deck != deck_name
    ]
    if cards_to_move:
        try:
            invoke("changeDeck", cards=cards_to_move, deck=deck_name)
            for cid in cards_to_move:
                nid = card_to_note[cid]
                logger.info(
                    f"  Moved card {cid} (note {nid}) "
                    f"from '{card_to_deck[cid]}' to '{deck_name}'"
                )
        except Exception as e:
            for cid in cards_to_move:
                result.errors.append(f"Card {cid}: failed to move deck: {e}")

    note_ids = list(set(card_to_note.values()))
    try:
        notes_info = invoke("notesInfo", notes=note_ids)
    except Exception as e:
        for card, _ in existing_cards:
            result.errors.append(f"Card {card.card_id} (line {card.line_number}): {e}")
        return []

    note_fields: dict[int, dict[str, str]] = {}
    for note in notes_info:
        if note:
            note_fields[note["noteId"]] = {
                name: info["value"] for name, info in note["fields"].items()
            }

    updates: list[dict] = []
    update_cards: list[ParsedCard] = []
    stale: list[tuple[ParsedCard, dict[str, str]]] = []

    for card, html_fields in existing_cards:
        assert card.card_id is not None
        note_id = card_to_note.get(card.card_id)
        if note_id is None:
            logger.info(
                f"  Card {card.card_id} (line {card.line_number}) "
                f"no longer in Anki, will re-create"
            )
            stale.append((card, html_fields))
            continue

        current = note_fields.get(note_id, {})
        if all(current.get(name) == value for name, value in html_fields.items()):
            result.skipped += 1
            continue

        updates.append(
            {
                "action": "updateNoteFields",
                "params": {"note": {"id": note_id, "fields": html_fields}},
            }
        )
        update_cards.append(card)

    if updates:
        try:
            multi_results = invoke("multi", actions=updates)
            for i, res in enumerate(multi_results):
                if res is None:
                    result.updated += 1
                else:
                    result.errors.append(
                        f"Card {update_cards[i].card_id} "
                        f"(line {update_cards[i].line_number}): "
                        f"{res}"
                    )
        except Exception as e:
            for card in update_cards:
                result.errors.append(
                    f"Card {card.card_id} (line {card.line_number}): {e}"
                )

    return stale


def _import_existing_cloze_notes(
    existing_cloze: list[tuple[ParsedCard, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedCard, dict[str, str]]]:
    """Phase 1b: Batch update existing DeckOpsCloze notes.

    Returns stale entries (note_id no longer in Anki) for re-creation.
    Also moves cards to the correct deck if they were moved between files.
    """
    if not existing_cloze:
        return []

    existing_note_ids = [card.note_id for card, _ in existing_cloze]

    try:
        notes_info = invoke("notesInfo", notes=existing_note_ids)
    except Exception as e:
        for card, _ in existing_cloze:
            result.errors.append(f"Note {card.note_id} (line {card.line_number}): {e}")
        return []

    # Safety check: Ensure all notes have the correct DeckOps template
    for note in notes_info:
        if note and note.get("modelName") and note["modelName"] != "DeckOpsCloze":
            raise ValueError(
                f"Safety check failed: Note {note['noteId']} has template "
                f"'{note['modelName']}' but expected 'DeckOpsCloze'. "
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
                # Find note IDs for logging
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
            for card, _ in existing_cloze:
                result.errors.append(f"Note {card.note_id}: failed to move deck: {e}")

    updates: list[dict] = []
    update_cards: list[ParsedCard] = []
    stale: list[tuple[ParsedCard, dict[str, str]]] = []

    for card, html_fields in existing_cloze:
        assert card.note_id is not None
        current = note_fields.get(card.note_id, {})
        if not current:
            logger.info(
                f"  Note {card.note_id} (line {card.line_number}) "
                f"no longer in Anki, will re-create"
            )
            stale.append((card, html_fields))
            continue

        if all(current.get(name) == value for name, value in html_fields.items()):
            result.skipped += 1
            continue

        updates.append(
            {
                "action": "updateNoteFields",
                "params": {"note": {"id": card.note_id, "fields": html_fields}},
            }
        )
        update_cards.append(card)

    if updates:
        try:
            multi_results = invoke("multi", actions=updates)
            for i, res in enumerate(multi_results):
                if res is None:
                    result.updated += 1
                else:
                    result.errors.append(
                        f"Note {update_cards[i].note_id} "
                        f"(line {update_cards[i].line_number}): "
                        f"{res}"
                    )
        except Exception as e:
            for card in update_cards:
                result.errors.append(
                    f"Note {card.note_id} (line {card.line_number}): {e}"
                )

    return stale


def _create_new_qa_cards(
    new_cards: list[tuple[ParsedCard, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedCard, int]]:
    """Phase 2: Batch create new DeckOpsQA cards. Returns (card, card_id) pairs."""
    if not new_cards:
        return []

    create_actions = [
        {
            "action": "addNote",
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "DeckOpsQA",
                    "fields": html_fields,
                    "options": {"allowDuplicate": False},
                }
            },
        }
        for _, html_fields in new_cards
    ]

    try:
        create_results = invoke("multi", actions=create_actions)
    except Exception as e:
        for card, _ in new_cards:
            result.errors.append(f"Card new (line {card.line_number}): {e}")
        return []

    successful: list[tuple[ParsedCard, int]] = []
    for i, note_id in enumerate(create_results):
        card, _ = new_cards[i]
        if note_id and isinstance(note_id, int):
            successful.append((card, note_id))
            result.created += 1
        else:
            result.errors.append(f"Card new (line {card.line_number}): {note_id}")

    # Resolve note_ids → card_ids for writing back
    card_id_assignments: list[tuple[ParsedCard, int]] = []
    if successful:
        new_note_ids = [nid for _, nid in successful]
        try:
            new_notes_info = invoke("notesInfo", notes=new_note_ids)
            for j, note_info in enumerate(new_notes_info):
                card, _ = successful[j]
                if note_info and note_info.get("cards"):
                    card_id_assignments.append((card, note_info["cards"][0]))
        except Exception as e:
            for card, _ in successful:
                result.errors.append(
                    f"Card new (line {card.line_number}): failed to fetch card_id: {e}"
                )

    return card_id_assignments


def _create_new_cloze_notes(
    new_cloze: list[tuple[ParsedCard, dict[str, str]]],
    deck_name: str,
    result: FileImportResult,
) -> list[tuple[ParsedCard, int]]:
    """Phase 2b: Batch create new DeckOpsCloze notes. Returns (card, note_id) pairs."""
    if not new_cloze:
        return []

    create_actions = [
        {
            "action": "addNote",
            "params": {
                "note": {
                    "deckName": deck_name,
                    "modelName": "DeckOpsCloze",
                    "fields": html_fields,
                    "options": {"allowDuplicate": False},
                }
            },
        }
        for _, html_fields in new_cloze
    ]

    try:
        create_results = invoke("multi", actions=create_actions)
    except Exception as e:
        for card, _ in new_cloze:
            result.errors.append(f"Note new (line {card.line_number}): {e}")
        return []

    note_id_assignments: list[tuple[ParsedCard, int]] = []
    for i, note_id in enumerate(create_results):
        card, _ = new_cloze[i]
        if note_id and isinstance(note_id, int):
            note_id_assignments.append((card, note_id))
            result.created += 1
        else:
            result.errors.append(f"Note new (line {card.line_number}): {note_id}")

    return note_id_assignments


def _delete_orphaned_notes(
    cards: list[ParsedCard],
    deck_name: str,
    result: FileImportResult,
    global_card_ids: set[int] | None = None,
    global_note_ids: set[int] | None = None,
) -> None:
    """Delete notes from Anki that are no longer in the markdown file.
    (cleanup within tracked files, card-level)

    Must run BEFORE creating new notes, so newly created notes
    can't be mistaken for orphans.

    When global_card_ids/global_note_ids are provided (collection mode),
    cards claimed by other files are excluded from deletion — those files
    will move the cards to the correct deck when they are processed.
    """
    # --- DeckOpsQA orphans (by card_id) ---
    md_card_ids = {
        c.card_id for c in cards if c.note_type == "DeckOpsQA" and c.card_id is not None
    }
    qa_query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:DeckOpsQA'
    anki_qa_card_ids = set(invoke("findCards", query=qa_query))
    orphaned_qa_cards = anki_qa_card_ids - md_card_ids
    if global_card_ids:
        orphaned_qa_cards -= global_card_ids

    if orphaned_qa_cards:
        cards_info = invoke("cardsInfo", cards=list(orphaned_qa_cards))
        # Build note_id → card_ids mapping for clear logging
        note_to_cards: dict[int, list[int]] = {}
        for c in cards_info:
            note_to_cards.setdefault(c["note"], []).append(c["cardId"])
        note_ids = list(note_to_cards.keys())
        invoke("deleteNotes", notes=note_ids)
        result.deleted += len(note_ids)
        for nid, cids in note_to_cards.items():
            cid_str = ", ".join(str(c) for c in cids)
            logger.info(f"  Deleted QA note {nid} (card {cid_str}) from Anki")

    # --- DeckOpsCloze orphans (by note_id) ---
    # Note that it is not possible to remove individual cloze cards from a note as of
    # early 2026 via AnkiConnect.
    md_note_ids = {
        c.note_id
        for c in cards
        if c.note_type == "DeckOpsCloze" and c.note_id is not None
    }
    cloze_query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:DeckOpsCloze'
    anki_cloze_card_ids = invoke("findCards", query=cloze_query)

    if anki_cloze_card_ids:
        cloze_cards_info = invoke("cardsInfo", cards=anki_cloze_card_ids)
        anki_cloze_note_ids = {c["note"] for c in cloze_cards_info}
        orphaned_cloze_notes = anki_cloze_note_ids - md_note_ids
        if global_note_ids:
            orphaned_cloze_notes -= global_note_ids

        if orphaned_cloze_notes:
            invoke("deleteNotes", notes=list(orphaned_cloze_notes))
            result.deleted += len(orphaned_cloze_notes)
            # Build note_id → card_ids mapping for clear logging
            note_to_cloze_cards: dict[int, list[int]] = {}
            for c in cloze_cards_info:
                if c["note"] in orphaned_cloze_notes:
                    note_to_cloze_cards.setdefault(c["note"], []).append(c["cardId"])
            for nid in orphaned_cloze_notes:
                cids = note_to_cloze_cards.get(nid, [])
                cid_str = ", ".join(str(c) for c in cids)
                logger.info(
                    f"  Deleted Cloze note {nid}"
                    f"{f' (cards {cid_str})' if cid_str else ''}"
                    f" from Anki"
                )


def import_file(
    file_path: Path,
    deck_name: str | None = None,
    only_add_new: bool = False,
    deck_names_and_ids: dict[str, int] | None = None,
    global_card_ids: set[int] | None = None,
    global_note_ids: set[int] | None = None,
) -> FileImportResult:
    """Import all cards from a markdown file into Anki.

    Handles both DeckOpsQA (card_id) and DeckOpsCloze (note_id) blocks.

    When global_card_ids/global_note_ids are provided, orphan deletion
    will not remove cards that are tracked by other files in the
    collection — they will be moved to the correct deck instead.
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

    cards = parse_markdown_file(file_path)
    converter = MarkdownToHTML()

    result = FileImportResult(
        file_path=file_path,
        total_notes=len(cards),
        updated=0,
        created=0,
        deleted=0,
        skipped=0,
    )

    # Phase 0: Classify cards by note type and convert fields
    existing_qa: list[tuple[ParsedCard, dict[str, str]]] = []
    new_qa: list[tuple[ParsedCard, dict[str, str]]] = []
    existing_cloze: list[tuple[ParsedCard, dict[str, str]]] = []
    new_cloze: list[tuple[ParsedCard, dict[str, str]]] = []

    for card in cards:
        validation_errors = validate_card(card)
        if validation_errors:
            for err in validation_errors:
                result.errors.append(
                    f"Card {card.card_id or card.note_id or 'new'} "
                    f"(line {card.line_number}): {err}"
                )
            continue

        try:
            html_fields = convert_fields_to_html(card.fields, converter)
        except Exception as e:
            result.errors.append(
                f"Card {card.card_id or card.note_id or 'new'} "
                f"(line {card.line_number}): {e}"
            )
            continue

        if card.note_type == "DeckOpsQA":
            if card.card_id:
                if only_add_new:
                    result.skipped += 1
                    continue
                existing_qa.append((card, html_fields))
            elif deck_name:
                new_qa.append((card, html_fields))
            else:
                result.errors.append(
                    f"Card new (line {card.line_number}): No deck_name"
                )

        elif card.note_type == "DeckOpsCloze":
            if card.note_id:
                if only_add_new:
                    result.skipped += 1
                    continue
                existing_cloze.append((card, html_fields))
            elif deck_name:
                new_cloze.append((card, html_fields))
            else:
                result.errors.append(
                    f"Note new (line {card.line_number}): No deck_name"
                )

    # Phase 1: Update existing DeckOpsQA cards (returns stale entries)
    stale_qa = _import_existing_qa_cards(existing_qa, deck_name, result)
    new_qa.extend(stale_qa)

    # Phase 1b: Update existing DeckOpsCloze notes (returns stale entries)
    stale_cloze = _import_existing_cloze_notes(existing_cloze, deck_name, result)
    new_cloze.extend(stale_cloze)

    # Phase 2: Delete orphaned notes
    # (BEFORE creation so new notes aren't mistaken for orphans)
    if deck_name:
        _delete_orphaned_notes(
            cards,
            deck_name,
            result,
            global_card_ids=global_card_ids,
            global_note_ids=global_note_ids,
        )

    # Phase 3: Create new DeckOpsQA cards (includes stale re-creations)
    qa_id_assignments = _create_new_qa_cards(new_qa, deck_name, result)

    # Phase 3b: Create new DeckOpsCloze notes (includes stale re-creations)
    cloze_id_assignments = _create_new_cloze_notes(new_cloze, deck_name, result)

    # Phase 4: Write IDs back to file
    all_id_assignments = qa_id_assignments + cloze_id_assignments
    _batch_write_ids_to_file(file_path, all_id_assignments)

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
    # This prevents orphan deletion from removing cards that were
    # moved between files — the claiming file will move them instead.
    # Also detect duplicate IDs across files and abort if found.
    md_files = sorted(collection_path.glob("*.md"))
    global_card_ids: set[int] = set()
    global_note_ids: set[int] = set()
    card_id_sources: dict[int, str] = {}  # card_id -> filename
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
        for card in parse_markdown_file(md_file):
            if card.card_id is not None:
                if card.card_id in card_id_sources:
                    duplicates.append(
                        f"Duplicate card_id {card.card_id} found in "
                        f"'{md_file.name}' and '{card_id_sources[card.card_id]}'"
                    )
                    duplicate_ids.add(card.card_id)
                else:
                    card_id_sources[card.card_id] = md_file.name
                global_card_ids.add(card.card_id)
            if card.note_id is not None:
                if card.note_id in note_id_sources:
                    duplicates.append(
                        f"Duplicate note_id {card.note_id} found in "
                        f"'{md_file.name}' and '{note_id_sources[card.note_id]}'"
                    )
                    duplicate_ids.add(card.note_id)
                else:
                    note_id_sources[card.note_id] = md_file.name
                global_note_ids.add(card.note_id)
    if duplicates:
        for dup in duplicates:
            logger.error(dup)
        ids_str = ", ".join(str(i) for i in sorted(duplicate_ids))
        raise ValueError(
            f"Aborting import: {len(duplicate_ids)} duplicate ID(s) found across files:"
            f" {ids_str}. Each deck/note/card ID must appear in exactly one file."
        )

    for md_file in md_files:
        logger.info(f"Processing {md_file.name}...")
        result = import_file(
            md_file,
            only_add_new=only_add_new,
            deck_names_and_ids=deck_names_and_ids,
            global_card_ids=global_card_ids,
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

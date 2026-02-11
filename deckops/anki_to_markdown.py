"""Transcribe Anki decks to Markdown files."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from deckops.anki_client import extract_deck_id, invoke
from deckops.config import CARD_SEPARATOR, SUPPORTED_NOTE_TYPES
from deckops.html_converter import HTMLToMarkdown
from deckops.markdown_helpers import (
    extract_card_blocks,
    format_card,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class DeckExportResult:
    """Result of exporting a single deck."""

    deck_name: str
    file_path: Path | None
    total_notes: int
    updated: int
    created: int
    deleted: int
    skipped: int
    # Block IDs (e.g. "card_id: 123") that appeared/disappeared
    # compared to the previous file, used for cross-deck move detection.
    created_ids: set[str] | None = None
    deleted_ids: set[str] | None = None


def transcribe_deck(
    deck_name: str, output_dir: str = ".", deck_id: int | None = None
) -> DeckExportResult:
    """Transcribe an Anki deck to a Markdown file (excluding subdecks)."""
    converter = HTMLToMarkdown()
    # Collect (note_id, formatted_block) tuples.
    # In Anki the note ID is the creation timestamp in milliseconds.
    blocks_with_ids: list[tuple[int, str]] = []

    # --- DeckOpsQA: one block per card ---
    qa_query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:DeckOpsQA'
    qa_card_ids = invoke("findCards", query=qa_query)

    if qa_card_ids:
        qa_cards_info = invoke("cardsInfo", cards=qa_card_ids)
        qa_note_ids = list({card["note"] for card in qa_cards_info})
        qa_notes_info = invoke("notesInfo", notes=qa_note_ids)
        qa_note_dict = {note["noteId"]: note for note in qa_notes_info}

        for card in qa_cards_info:
            blocks_with_ids.append(
                (
                    card["note"],
                    format_card(
                        card["cardId"],
                        qa_note_dict[card["note"]],
                        converter,
                        note_type="DeckOpsQA",
                    ),
                )
            )

    # --- DeckOpsCloze: one block per note (deduplicated) ---
    cloze_query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:DeckOpsCloze'
    cloze_card_ids = invoke("findCards", query=cloze_query)

    if cloze_card_ids:
        cloze_cards_info = invoke("cardsInfo", cards=cloze_card_ids)
        cloze_note_ids = list({card["note"] for card in cloze_cards_info})
        cloze_notes_info = invoke("notesInfo", notes=cloze_note_ids)

        for note in cloze_notes_info:
            blocks_with_ids.append(
                (
                    note["noteId"],
                    format_card(
                        note["noteId"],
                        note,
                        converter,
                        note_type="DeckOpsCloze",
                    ),
                )
            )

    # Build a lookup from block ID string to (note_id, formatted_block)
    block_by_id: dict[str, tuple[int, str]] = {}
    for note_id, block in blocks_with_ids:
        match = re.match(r"<!--\s*((?:card_id|note_id):\s*\d+)\s*-->", block)
        if match:
            key = re.sub(r"\s+", " ", match.group(1))
            block_by_id[key] = (note_id, block)

    if not blocks_with_ids:
        return DeckExportResult(
            deck_name=deck_name,
            file_path=None,
            total_notes=0,
            updated=0,
            created=0,
            deleted=0,
            skipped=0,
        )

    output_path = Path(output_dir) / (sanitize_filename(deck_name) + ".md")
    deck_id_line = "<!-- deck_id: {} -->".format(deck_id) + "\n" if deck_id else ""

    # Compare with existing file to determine per-card changes
    updated = 0
    created = 0
    deleted = 0
    skipped = 0
    created_ids: set[str] = set()
    deleted_ids: set[str] = set()

    old_content = (
        output_path.read_text(encoding="utf-8") if output_path.exists() else None
    )

    if old_content is not None:
        existing_blocks = extract_card_blocks(old_content)

        # Preserve existing file order; append new cards sorted by creation date.
        new_block_ids = set(block_by_id.keys())
        ordered_blocks: list[str] = []

        # Keep existing cards in their current order, updating content
        for block_id in existing_blocks:
            if block_id in block_by_id:
                _, block = block_by_id[block_id]
                ordered_blocks.append(block)
                if existing_blocks[block_id] == block:
                    skipped += 1
                else:
                    updated += 1
            else:
                deleted += 1
                deleted_ids.add(block_id)

        # Append genuinely new cards, sorted by creation date among themselves
        new_ids = new_block_ids - set(existing_blocks)
        new_entries = sorted(
            ((bid, *block_by_id[bid]) for bid in new_ids),
            key=lambda x: x[1],  # sort by note_id (creation timestamp)
        )
        for bid, _, block in new_entries:
            ordered_blocks.append(block)
            created += 1
            created_ids.add(bid)

        markdown_blocks = ordered_blocks
    else:
        # First export for this deck: sort by creation date (chronological).
        blocks_with_ids.sort(key=lambda x: x[0])
        markdown_blocks = [block for _, block in blocks_with_ids]
        created = len(markdown_blocks)

    cards_content = CARD_SEPARATOR.join(markdown_blocks)
    new_content = deck_id_line + cards_content

    # Only write if content actually changed
    if old_content != new_content:
        output_path.write_text(new_content, encoding="utf-8")

    logger.debug(f"{deck_name}: {len(markdown_blocks)} blocks -> {output_path.name}")
    return DeckExportResult(
        deck_name=deck_name,
        file_path=output_path,
        total_notes=len(markdown_blocks),
        updated=updated,
        created=created,
        deleted=deleted,
        skipped=skipped,
        created_ids=created_ids,
        deleted_ids=deleted_ids,
    )


def _find_relevant_decks() -> set[str]:
    """Return deck names that contain DeckOpsQA or DeckOpsCloze notes."""
    query = " OR ".join(f"note:{nt}" for nt in SUPPORTED_NOTE_TYPES)
    card_ids = invoke("findCards", query=query)
    if not card_ids:
        return set()
    cards_info = invoke("cardsInfo", cards=card_ids)
    return {card["deckName"] for card in cards_info}


def transcribe_collection(output_dir: str = ".") -> list[DeckExportResult]:
    """Transcribe all decks in the collection to Markdown files."""
    deck_names_and_ids = invoke("deckNamesAndIds")
    relevant_decks = _find_relevant_decks()
    logger.info(
        f"Found {len(deck_names_and_ids)} decks, "
        f"{len(relevant_decks)} with supported note types"
    )

    # Also include decks that have existing markdown files (they may
    # have become empty after cards moved out and need updating).
    output_path = Path(output_dir)
    id_to_name = {v: k for k, v in deck_names_and_ids.items()}
    for md_file in output_path.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        deck_id, _ = extract_deck_id(content)
        if deck_id and deck_id in id_to_name:
            relevant_decks.add(id_to_name[deck_id])

    results = []
    for deck_name in sorted(deck_names_and_ids):
        if deck_name not in relevant_decks:
            continue
        deck_id = deck_names_and_ids[deck_name]
        logger.info(f"Processing {deck_name} (id: {deck_id})...")
        result = transcribe_deck(deck_name, output_dir, deck_id=deck_id)
        if result.file_path:
            results.append(result)

        if result.total_notes > 0:
            logger.info(
                f"  Updated: {result.updated}, Created: {result.created}, "
                f"Deleted: {result.deleted}, Skipped: {result.skipped}"
            )

    # Check for cross-deck moves: a card/note ID that disappeared from
    # one file and appeared in another was moved between decks in Anki.
    all_created_ids: set[str] = set()
    all_deleted_ids: set[str] = set()
    for r in results:
        all_created_ids.update(r.created_ids or set())
        all_deleted_ids.update(r.deleted_ids or set())
    moved = len(all_created_ids & all_deleted_ids)
    if moved:
        logger.info(
            f"  Note: {moved} of the above created/deleted card(s) were "
            f"moved between decks (review history is preserved)"
        )

    return results


def rename_markdown_files(output_dir: str = ".") -> int:
    """Rename markdown files to match their Anki deck name via deck_id.

    If a deck was renamed in Anki, the corresponding markdown file is
    renamed to reflect the new deck name.  The deck_id inside the file
    is used to link the file to its deck.
    Returns the number of renamed files.
    """
    deck_names_and_ids = invoke("deckNamesAndIds")
    id_to_name = {v: k for k, v in deck_names_and_ids.items()}
    renamed = 0

    for md_file in Path(output_dir).glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        deck_id, _ = extract_deck_id(content)
        if deck_id is None or deck_id not in id_to_name:
            continue

        expected_filename = sanitize_filename(id_to_name[deck_id]) + ".md"
        if md_file.name != expected_filename:
            new_path = md_file.parent / expected_filename
            logger.info(f"  Renamed {md_file.name} -> {expected_filename}")
            md_file.rename(new_path)
            renamed += 1

    return renamed


def delete_orphaned_decks(output_dir: str = ".") -> int:
    """Delete markdown files whose deck_id is not found in Anki.

    Files without a deck_id are kept (they are likely new decks pending first sync).
    Returns the number of deleted files.
    """
    anki_deck_ids = set(invoke("deckNamesAndIds").values())
    deleted = 0

    for md_file in Path(output_dir).glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        deck_id, _ = extract_deck_id(content)
        if deck_id is not None and deck_id not in anki_deck_ids:
            logger.info(
                f"  Deleting orphaned deck file {md_file.name} (deck_id: {deck_id})"
            )
            md_file.unlink()
            deleted += 1

    return deleted


def delete_orphaned_cards(output_dir: str = ".") -> int:
    """Delete cards/notes from markdown files whose IDs are not found in Anki.

    Cards without an ID are kept (they are new cards pending first sync).
    Returns the total number of deleted blocks.
    """
    anki_card_ids = set(invoke("findCards", query="deck:*"))
    # Get all note IDs for note_id-based blocks (DeckOpsCloze)
    anki_note_ids: set[int] = set()
    for note_type in SUPPORTED_NOTE_TYPES:
        note_ids = invoke("findNotes", query=f"note:{note_type}")
        anki_note_ids.update(note_ids)

    total_deleted = 0

    for md_file in Path(output_dir).glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        deck_id_line_match = re.match(r"(<!--\s*deck_id:\s*\d+\s*-->\n?)", content)
        deck_id_prefix = deck_id_line_match.group(1) if deck_id_line_match else ""
        _, cards_content = extract_deck_id(content)

        blocks = cards_content.split(CARD_SEPARATOR)
        kept: list[str] = []
        deleted = 0

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                continue

            card_match = re.match(r"<!--\s*card_id:\s*(\d+)\s*-->", stripped)
            if card_match:
                card_id = int(card_match.group(1))
                if card_id not in anki_card_ids:
                    deleted += 1
                    logger.info(f"  Deleting card {card_id} from {md_file.name}")
                    continue

            note_match = re.match(r"<!--\s*note_id:\s*(\d+)\s*-->", stripped)
            if note_match:
                note_id = int(note_match.group(1))
                if note_id not in anki_note_ids:
                    deleted += 1
                    logger.info(f"  Deleting note {note_id} from {md_file.name}")
                    continue

            kept.append(stripped)

        if deleted > 0:
            new_content = deck_id_prefix + CARD_SEPARATOR.join(kept)
            md_file.write_text(new_content, encoding="utf-8")
            logger.info(f"{md_file.name}: deleted {deleted} orphaned block(s)")
            total_deleted += deleted

    return total_deleted

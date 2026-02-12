"""Transcribe Anki decks to Markdown files."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from deckops.anki_client import invoke
from deckops.config import NOTE_SEPARATOR, SUPPORTED_NOTE_TYPES
from deckops.html_converter import HTMLToMarkdown
from deckops.markdown_helpers import (
    extract_deck_id,
    extract_note_blocks,
    format_note,
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
    # Block IDs (e.g. "note_id: 123") that appeared/disappeared
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

    for note_type in SUPPORTED_NOTE_TYPES:
        query = f'deck:"{deck_name}" -deck:"{deck_name}::*" note:{note_type}'
        card_ids = invoke("findCards", query=query)

        if not card_ids:
            continue

        cards_info = invoke("cardsInfo", cards=card_ids)
        note_ids = list({card["note"] for card in cards_info})
        notes_info = invoke("notesInfo", notes=note_ids)

        for note in notes_info:
            blocks_with_ids.append(
                (
                    note["noteId"],
                    format_note(
                        note["noteId"],
                        note,
                        converter,
                        note_type=note_type,
                    ),
                )
            )

    # Build a lookup from block ID string to (note_id, formatted_block)
    block_by_id: dict[str, tuple[int, str]] = {}
    for note_id, block in blocks_with_ids:
        match = re.match(r"<!--\s*(note_id:\s*\d+)\s*-->", block)
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

    # Compare with existing file to determine per-note changes
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
        existing_blocks = extract_note_blocks(old_content)

        # Preserve existing file order; append new notes sorted by creation date.
        new_block_ids = set(block_by_id.keys())
        ordered_blocks: list[str] = []

        # Keep existing notes in their current order, updating content
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

        # Append genuinely new notes, sorted by creation date among themselves
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

    notes_content = NOTE_SEPARATOR.join(markdown_blocks)
    new_content = deck_id_line + notes_content

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

    # Don't count the "default" deck if empty (has no cards)
    # and there are other decks in the collection
    total_decks = len(deck_names_and_ids)
    if total_decks > 1:
        default_card_ids = invoke(
            "findCards", query='deck:"default" -deck:"default::*"'
        )
        if not default_card_ids:
            total_decks -= 1

    logger.info(
        f"Found {total_decks} decks, {len(relevant_decks)} with supported note types"
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

    # Check for cross-deck moves: a note ID that disappeared from
    # one file and appeared in another was moved between decks in Anki.
    all_created_ids: set[str] = set()
    all_deleted_ids: set[str] = set()
    for r in results:
        all_created_ids.update(r.created_ids or set())
        all_deleted_ids.update(r.deleted_ids or set())
    moved = len(all_created_ids & all_deleted_ids)
    if moved:
        logger.info(
            f"  Note: {moved} of the above created/deleted note(s) were "
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


def delete_orphaned_notes(output_dir: str = ".") -> int:
    """Delete notes from markdown files whose IDs are not found in Anki.

    Notes without an ID are kept (they are new notes pending first sync).
    Returns the total number of deleted blocks.
    """
    # Get all note IDs across all supported note types
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

        blocks = cards_content.split(NOTE_SEPARATOR)
        kept: list[str] = []
        deleted = 0

        for block in blocks:
            stripped = block.strip()
            if not stripped:
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
            new_content = deck_id_prefix + NOTE_SEPARATOR.join(kept)
            md_file.write_text(new_content, encoding="utf-8")
            logger.info(f"{md_file.name}: deleted {deleted} orphaned block(s)")
            total_deleted += deleted

    return total_deleted

"""Export Anki decks to Markdown files.

Architecture:
  AnkiState   – all Anki-side data, fetched once (shared from anki_client)
  FileState   – one existing markdown file, read once (from models)
  _sync_deck  – single engine: diff existing file vs Anki state, return new content
  export_collection – orchestrates: rename → sync → delete orphans (one pass)
"""

import logging
import re
from pathlib import Path

from ankiops.config import NOTE_SEPARATOR, SUPPORTED_NOTE_TYPES, sanitize_filename
from ankiops.html_converter import HTMLToMarkdown
from ankiops.log import clickable_path, format_changes
from ankiops.models import AnkiState, ExportSummary, FileState, SyncResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


def _format_blocks(
    note_ids: set[int],
    anki: AnkiState,
    converter: HTMLToMarkdown,
) -> dict[str, tuple[int, str]]:
    """Format Anki notes into markdown blocks.

    Returns {block_id_key: (note_id, formatted_block)}.
    """
    block_by_id: dict[str, tuple[int, str]] = {}

    for nid in note_ids:
        anki_note = anki.notes.get(nid)
        if not anki_note:
            continue
        if anki_note.note_type not in SUPPORTED_NOTE_TYPES:
            continue
        block = anki_note.to_markdown(converter)
        match = re.match(r"<!--\s*(note_id:\s*\d+)\s*-->", block)
        if match:
            key = re.sub(r"\s+", " ", match.group(1))
            block_by_id[key] = (nid, block)

    return block_by_id


def _sync_deck(
    deck_name: str,
    deck_id: int,
    anki: AnkiState,
    converter: HTMLToMarkdown,
    existing_file: FileState | None,
) -> tuple[SyncResult, str | None]:
    """Synchronize one Anki deck to markdown content.

    Returns (result, new_content). new_content is None if the deck
    is empty (no notes to export).
    """
    note_ids = anki.deck_note_ids.get(deck_name, set())
    block_by_id = _format_blocks(note_ids, anki, converter)

    if not block_by_id:
        return SyncResult(
            deck_name=deck_name,
            file_path=None,
            total_notes=0,
            updated=0,
            created=0,
            deleted=0,
            moved=0,
            skipped=0,
        ), None

    deck_id_line = f"<!-- deck_id: {deck_id} -->\n"

    updated = 0
    created = 0
    deleted = 0
    skipped = 0
    moved = 0

    if existing_file is not None:
        existing_blocks = existing_file.existing_blocks
        new_block_ids = set(block_by_id.keys())
        ordered_blocks: list[str] = []

        # Preserve existing order, updating content
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

        # Append genuinely new notes, sorted by creation date
        new_ids = new_block_ids - set(existing_blocks)
        new_entries = sorted(
            ((bid, *block_by_id[bid]) for bid in new_ids),
            key=lambda x: x[1],  # sort by note_id (creation timestamp)
        )
        for _, _, block in new_entries:
            ordered_blocks.append(block)
            created += 1

        markdown_blocks = ordered_blocks
    else:
        # First export: sort by creation date
        sorted_blocks = sorted(block_by_id.values(), key=lambda x: x[0])
        markdown_blocks = [block for _, block in sorted_blocks]
        created = len(markdown_blocks)

    new_content = deck_id_line + NOTE_SEPARATOR.join(markdown_blocks)

    result = SyncResult(
        deck_name=deck_name,
        file_path=None,  # Set by caller after writing
        total_notes=len(markdown_blocks),
        updated=updated,
        created=created,
        deleted=deleted,
        moved=moved,
        skipped=skipped,
    )
    return result, new_content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_deck(
    deck_name: str,
    output_dir: str = ".",
    deck_id: int | None = None,
) -> SyncResult:
    """Export a single Anki deck to a Markdown file."""
    anki = AnkiState.fetch()
    converter = HTMLToMarkdown()

    if deck_id is None:
        deck_id = anki.deck_names_and_ids.get(deck_name)
    if deck_id is None:
        raise ValueError(f"Deck '{deck_name}' not found in Anki")

    output_path = Path(output_dir) / (sanitize_filename(deck_name) + ".md")
    existing_file = FileState.from_file(output_path) if output_path.exists() else None

    # Check for untracked notes before overwriting
    if existing_file and existing_file.has_untracked:
        logger.warning(
            f"The file {output_path.name} contains new notes without note IDs."
        )
        logger.warning(
            "\nThese notes have not been imported to Anki yet and will be LOST "
            "if you continue with the export."
        )
        logger.warning(
            "\nTo preserve them, first run:\n  ankiops markdown-to-anki --only-add-new"
        )
        answer = (
            input("\nContinue with export anyway (new notes will be lost)? [y/N] ")
            .strip()
            .lower()
        )
        if answer != "y":
            logger.info("Export cancelled. Import your new notes first.")
            raise SystemExit(0)

    result, new_content = _sync_deck(deck_name, deck_id, anki, converter, existing_file)

    if new_content is not None:
        old_content = existing_file.raw_content if existing_file else None
        if old_content != new_content:
            output_path.write_text(new_content, encoding="utf-8")
        result.file_path = output_path

    return result


def export_collection(
    output_dir: str = ".",
    keep_orphans: bool = False,
) -> ExportSummary:
    """Export all Anki decks to Markdown files in a single pass.

    Orchestrates the entire export:
      1. Fetch all Anki state (3-4 API calls)
      2. Read all existing markdown files (one read each)
      3. Rename files for decks renamed in Anki
      4. Sync each relevant deck
      5. Delete orphaned deck files (deck_id not in Anki)
      6. Delete orphaned notes (note_id not in Anki)

    Returns an ExportSummary with all results.
    """
    output_path = Path(output_dir)

    # Phase 1: Fetch all Anki state
    anki = AnkiState.fetch()
    converter = HTMLToMarkdown()
    all_note_ids = set(anki.notes.keys())

    # Phase 2: Read all existing markdown files
    files_by_deck_id: dict[int, FileState] = {}
    files_by_path: dict[Path, FileState] = {}
    unlinked_files: list[FileState] = []  # Files without a deck_id

    for md_file in output_path.glob("*.md"):
        fs = FileState.from_file(md_file)
        files_by_path[md_file] = fs
        if fs.deck_id is not None:
            files_by_deck_id[fs.deck_id] = fs
        else:
            unlinked_files.append(fs)

    # Check for untracked notes (notes without IDs) before overwriting
    files_with_untracked: list[Path] = []
    for md_file, fs in files_by_path.items():
        if fs.has_untracked:
            files_with_untracked.append(md_file)

    if files_with_untracked:
        logger.warning(
            "The following markdown files contain new notes without note IDs:"
        )
        for file_path in files_with_untracked:
            logger.warning(f"  - {clickable_path(file_path)}")
        logger.warning(
            "\nThese notes have not been imported to Anki yet and will be LOST "
            "if you continue with the export."
        )
        logger.warning(
            "\nTo preserve them, first run:\n  ankiops markdown-to-anki --only-add-new"
        )
        answer = (
            input("\nContinue with export anyway (new notes will be lost)? [y/N] ")
            .strip()
            .lower()
        )
        if answer != "y":
            logger.info("Export cancelled. Import your new notes first.")
            raise SystemExit(0)

    # Phase 3: Rename files for decks renamed in Anki
    renamed_files = 0
    for deck_id, fs in list(files_by_deck_id.items()):
        if deck_id not in anki.id_to_deck_name:
            continue
        expected_name = sanitize_filename(anki.id_to_deck_name[deck_id]) + ".md"
        if fs.file_path.name != expected_name:
            new_path = fs.file_path.parent / expected_name
            logger.info(
                f"Renamed {clickable_path(fs.file_path)} -> {clickable_path(new_path)}"
            )
            fs.file_path.rename(new_path)
            # Update references to the new path
            del files_by_path[fs.file_path]
            fs = FileState(
                file_path=new_path,
                raw_content=fs.raw_content,
                deck_id=fs.deck_id,
                parsed_notes=fs.parsed_notes,
            )
            files_by_deck_id[deck_id] = fs
            files_by_path[new_path] = fs
            renamed_files += 1

    # Phase 4: Determine relevant decks
    # A deck is relevant if it has AnkiOps notes OR has an existing file
    relevant_decks: set[str] = set()
    for deck_name in anki.deck_note_ids:
        relevant_decks.add(deck_name)
    for deck_id, fs in files_by_deck_id.items():
        if deck_id in anki.id_to_deck_name:
            relevant_decks.add(anki.id_to_deck_name[deck_id])

    # Log deck count (skip empty default deck)
    total_decks = len(anki.deck_names_and_ids)
    if total_decks > 1 and not anki.deck_note_ids.get("default"):
        total_decks -= 1
    logger.debug(
        f"Found {total_decks} decks, {len(relevant_decks)} with supported note types"
    )

    # Phase 5: Sync each relevant deck
    deck_results: list[SyncResult] = []
    all_created_ids: set[str] = set()
    all_deleted_ids: set[str] = set()

    for deck_name in sorted(anki.deck_names_and_ids):
        if deck_name not in relevant_decks:
            continue

        deck_id = anki.deck_names_and_ids[deck_name]
        existing_file = files_by_deck_id.get(deck_id)

        logger.debug(f"Processing {deck_name} (id: {deck_id})...")
        result, new_content = _sync_deck(
            deck_name, deck_id, anki, converter, existing_file
        )

        if new_content is not None:
            file_path = output_path / (sanitize_filename(deck_name) + ".md")
            old_content = existing_file.raw_content if existing_file else None
            if old_content != new_content:
                file_path.write_text(new_content, encoding="utf-8")
            result.file_path = file_path
            deck_results.append(result)

            # Track created/deleted IDs for move detection
            new_block_ids = set()
            for line in new_content.split("\n"):
                m = re.match(r"<!--\s*(note_id:\s*\d+)\s*-->", line)
                if m:
                    new_block_ids.add(re.sub(r"\s+", " ", m.group(1)))

            if existing_file:
                old_block_ids = set(existing_file.existing_blocks.keys())
                created_ids = new_block_ids - old_block_ids
                deleted_ids = old_block_ids - new_block_ids
                all_created_ids.update(created_ids)
                all_deleted_ids.update(deleted_ids)

        changes = format_changes(
            updated=result.updated,
            created=result.created,
            deleted=result.deleted,
        )
        if changes != "no changes":
            logger.info(f"  {clickable_path(result.file_path)}: {changes}")

    # Detect cross-deck moves
    moved_ids = all_created_ids & all_deleted_ids
    if moved_ids:
        logger.info(
            f"  {len(moved_ids)} note(s) moved between decks (review history preserved)"
        )

    # Phase 6: Delete orphaned deck files and notes
    deleted_deck_files = 0
    deleted_orphan_notes = 0

    if not keep_orphans:
        anki_deck_ids = set(anki.deck_names_and_ids.values())

        # Delete files whose deck_id doesn't exist in Anki
        for deck_id, fs in files_by_deck_id.items():
            if deck_id not in anki_deck_ids:
                logger.info(
                    f"Deleted orphaned deck file {clickable_path(fs.file_path)}"
                )
                fs.file_path.unlink()
                deleted_deck_files += 1

        # Delete notes from markdown files whose note_id isn't in Anki
        for md_file in output_path.glob("*.md"):
            # Re-read files that were just written (content may have changed)
            content = md_file.read_text(encoding="utf-8")
            deck_id_match = re.match(r"(<!--\s*deck_id:\s*\d+\s*-->\n?)", content)
            deck_id_prefix = deck_id_match.group(1) if deck_id_match else ""
            _, cards_content = FileState.extract_deck_id(content)

            blocks = cards_content.split(NOTE_SEPARATOR)
            kept: list[str] = []
            deleted = 0

            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    continue
                note_match = re.match(r"<!--\s*note_id:\s*(\d+)\s*-->", stripped)
                if note_match:
                    nid = int(note_match.group(1))
                    if nid not in all_note_ids:
                        deleted += 1
                        logger.debug(f"Deleting note {nid} from {md_file.name}")
                        continue
                kept.append(stripped)

            if deleted > 0:
                new_content = deck_id_prefix + NOTE_SEPARATOR.join(kept)
                md_file.write_text(new_content, encoding="utf-8")
                logger.info(f"  {md_file.name}: {deleted} orphaned note(s) deleted")
                deleted_orphan_notes += deleted

    return ExportSummary(
        deck_results=deck_results,
        renamed_files=renamed_files,
        deleted_deck_files=deleted_deck_files,
        deleted_orphan_notes=deleted_orphan_notes,
    )

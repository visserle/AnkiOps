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

from ankiops.config import NOTE_SEPARATOR, sanitize_filename
from ankiops.note_type_config import registry
from ankiops.html_converter import HTMLToMarkdown
from ankiops.log import clickable_path, format_changes
from ankiops.models import (
    AnkiState,
    Change,
    ChangeType,
    ExportSummary,
    FileState,
    SyncResult,
)

logger = logging.getLogger(__name__)


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
        anki_note = anki.notes_by_id.get(nid)
        if not anki_note:
            continue
        if anki_note.note_type not in registry.supported_note_types:
            continue
        block = anki_note.to_markdown(converter)
        match = re.match(r"<!--\s*(note_id:\s*\d+)\s*-->", block)
        if match:
            key = re.sub(r"\s+", " ", match.group(1))
            block_by_id[key] = (nid, block)

    return block_by_id


def _reconcile_blocks(
    block_by_id: dict[str, tuple[int, str]],
    existing_blocks: dict[str, str] | None,
) -> list[Change]:
    """Reconcile new Anki blocks with existing file blocks.

    Args:
        block_by_id: {block_id: (note_id, content)} from Anki.
        existing_blocks: {block_id: content} from existing file.

    Returns:
        List of Change objects representing the transformation.
    """
    changes: list[Change] = []

    if existing_blocks:
        new_block_ids = set(block_by_id.keys())

        # Preserve existing order, updating content
        for block_id, old_content in existing_blocks.items():
            if block_id in block_by_id:
                note_id, new_content = block_by_id[block_id]

                if old_content == new_content:
                    changes.append(
                        Change(
                            ChangeType.SKIP,
                            note_id,
                            block_id,
                            context={"block_content": new_content},
                        )
                    )
                else:
                    changes.append(
                        Change(
                            ChangeType.UPDATE,
                            note_id,
                            block_id,
                            context={"block_content": new_content},
                        )
                    )
            else:
                # Block exists in file but not in Anki -> Delete from file
                # Try to extract note_id from block_id ("note_id: 123")
                match = re.search(r"note_id:\s*(\d+)", block_id)
                entity_id = int(match.group(1)) if match else None

                changes.append(Change(ChangeType.DELETE, entity_id, block_id))

        # Append genuinely new notes, sorted by creation date
        new_ids = new_block_ids - set(existing_blocks)
        new_entries = sorted(
            ((bid, *block_by_id[bid]) for bid in new_ids),
            key=lambda x: x[1],  # sort by note_id (creation timestamp)
        )
        for bid, nid, content in new_entries:
            changes.append(
                Change(ChangeType.CREATE, nid, bid, context={"block_content": content})
            )

    else:
        # First export: sort by creation date (note_id)
        sorted_blocks = sorted(block_by_id.items(), key=lambda x: x[1][0])
        for bid, (nid, content) in sorted_blocks:
            changes.append(
                Change(ChangeType.CREATE, nid, bid, context={"block_content": content})
            )

    return changes


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
    note_ids = anki.note_ids_by_deck_name.get(deck_name, set())
    block_by_id = _format_blocks(note_ids, anki, converter)
    if not block_by_id:
        return SyncResult(
            deck_name=deck_name,
            file_path=None,
        ), None

    deck_id_line = f"<!-- deck_id: {deck_id} -->\n"
    existing_blocks = existing_file.existing_blocks if existing_file else None

    changes = _reconcile_blocks(block_by_id, existing_blocks)

    # Reconstruct content from changes
    new_blocks = []
    for change in changes:
        if change.change_type in (
            ChangeType.CREATE,
            ChangeType.UPDATE,
            ChangeType.SKIP,
        ):
            content = change.context.get("block_content")
            if content:
                new_blocks.append(content)

    new_content = deck_id_line + NOTE_SEPARATOR.join(new_blocks)

    result = SyncResult(
        deck_name=deck_name,
        file_path=None,  # Set by caller after writing
        changes=changes,
    )
    return result, new_content


def export_deck(
    deck_name: str,
    output_dir: str = ".",
    deck_id: int | None = None,
) -> SyncResult:
    """Export a single Anki deck to a Markdown file."""
    anki = AnkiState.fetch()
    converter = HTMLToMarkdown()

    if deck_id is None:
        deck_id = anki.deck_ids_by_name.get(deck_name)
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
        if deck_id not in anki.deck_names_by_id:
            continue
        expected_name = sanitize_filename(anki.deck_names_by_id[deck_id]) + ".md"
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
    for deck_name in anki.note_ids_by_deck_name:
        relevant_decks.add(deck_name)
    for deck_id, fs in files_by_deck_id.items():
        if deck_id in anki.deck_names_by_id:
            relevant_decks.add(anki.deck_names_by_id[deck_id])

    # Log deck count (skip empty default deck)
    total_decks = len(anki.deck_ids_by_name)
    if total_decks > 1 and not anki.note_ids_by_deck_name.get("default"):
        total_decks -= 1
    logger.debug(
        f"Found {total_decks} decks, {len(relevant_decks)} with supported note types"
    )

    # Phase 5: Sync each relevant deck
    deck_results: list[SyncResult] = []
    all_created_ids: set[str] = set()
    all_deleted_ids: set[str] = set()

    for deck_name in sorted(anki.deck_ids_by_name):
        if deck_name not in relevant_decks:
            continue

        deck_id = anki.deck_ids_by_name[deck_name]
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
            updated=result.updated_count,
            created=result.created_count,
            deleted=result.deleted_count,
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
        anki_deck_ids = set(anki.deck_ids_by_name.values())

        # Delete files whose deck_id doesn't exist in Anki
        for deck_id, fs in files_by_deck_id.items():
            if deck_id not in anki_deck_ids:
                logger.info(
                    f"Deleted orphaned deck file {clickable_path(fs.file_path)}"
                )
                fs.file_path.unlink()
                deleted_deck_files += 1

        # Note: Orphaned notes within valid deck files are already removed by _sync_deck
        # because _reconcile_blocks creates DELETE changes for them.
        # We only need to guard against files that weren't synced (e.g. unlinked),
        # but those are edge cases.

    return ExportSummary(
        deck_results=deck_results,
        renamed_files=renamed_files,
        deleted_deck_files=deleted_deck_files,
        deleted_orphan_notes=deleted_orphan_notes,
    )

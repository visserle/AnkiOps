"""Collection-level Markdown-to-Anki sync."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ankiops.anki import Anki
from ankiops.collection import file_stem_to_deck_name
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
)
from ankiops.markdown import DeckFile, read_deck_file
from ankiops.markdown_to_html import MarkdownToHTML
from ankiops.note_types import NoteType
from ankiops.sync.identity import resolve_import_note_identity
from ankiops.sync.report import (
    CollectionReport,
    ProtectedNoteGroup,
    UntrackedDeck,
)
from ankiops.sync.state import SyncState
from ankiops.sync.to_anki_deck import (
    PendingDeckWrite,
    collect_membership_affected_note_ids,
    flush_deck_metadata_writes,
    group_note_ids_by_deck_name,
    refresh_membership_for_affected_notes,
    sync_deck_to_anki,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SourceDeckFile:
    source: DeckSource
    file_state: DeckFile


def _load_source_deck_files(
    collection_dir: Path,
    note_types_dir: Path,
    sources: Sequence[DeckSource] | None = None,
) -> tuple[list[_SourceDeckFile], dict[str, NoteType]]:
    selected_sources = (
        list(sources)
        if sources is not None
        else discover_deck_sources(collection_dir, note_types_dir=note_types_dir)
    )
    note_types_by_name: dict[str, NoteType] = {}

    source_deck_files: list[_SourceDeckFile] = []
    for source in selected_sources:
        note_types = load_note_types_for_source(source)
        note_types_by_name.update(
            {note_type.name: note_type for note_type in note_types}
        )
        for deck_path in source.deck_files():
            source_deck_files.append(
                _SourceDeckFile(
                    source=source,
                    file_state=read_deck_file(
                        deck_path,
                        note_types=note_types,
                        context_root=source.root,
                    ),
                )
            )

    return source_deck_files, note_types_by_name


def _check_deck_ownership(source_deck_files: list[_SourceDeckFile]) -> None:
    deck_sources: dict[str, str] = {}
    duplicates: list[str] = []

    for source_deck_file in source_deck_files:
        deck_name = file_stem_to_deck_name(source_deck_file.file_state.file_path.stem)
        previous = deck_sources.get(deck_name)
        current = (
            f"{source_deck_file.source.display_name}:"
            f"{source_deck_file.file_state.file_path.name}"
        )
        if previous is not None:
            duplicates.append(
                f"Deck '{deck_name}' is defined by both {previous} and {current}"
            )
            continue
        deck_sources[deck_name] = current

    if duplicates:
        raise ValueError(
            "Aborting import: deck ownership conflicts found: " + "; ".join(duplicates)
        )


def _collect_global_note_keys(deck_files: list[DeckFile]) -> set[str]:
    note_key_locations: dict[str, str] = {}
    duplicates: list[str] = []

    for deck_file in deck_files:
        for note in deck_file.notes:
            if not note.note_key:
                continue
            if note.note_key in note_key_locations:
                duplicates.append(f"Duplicate note_key {note.note_key}")
            else:
                note_key_locations[note.note_key] = str(deck_file.file_path)

    if duplicates:
        raise ValueError(f"Aborting import: Duplicates found: {duplicates}")

    return set(note_key_locations)


def sync_collection_to_anki(
    anki_port: Anki,
    db_port: SyncState,
    collection_dir: Path,
    note_types_dir: Path,
    *,
    sources: Sequence[DeckSource] | None = None,
) -> CollectionReport:
    source_deck_files, note_types_by_name = _load_source_deck_files(
        collection_dir,
        note_types_dir,
        sources,
    )
    _check_deck_ownership(source_deck_files)

    deck_files = [source_deck_file.file_state for source_deck_file in source_deck_files]
    global_note_keys = _collect_global_note_keys(deck_files)
    required_note_types = sorted(note_types_by_name)
    markdown_to_html = MarkdownToHTML()

    note_import_fingerprints_by_note_key = db_port.resolve_import_hashes(
        global_note_keys
    )
    source_by_note_key = {
        note.note_key: source_deck_file.source.source_id
        for source_deck_file in source_deck_files
        for note in source_deck_file.file_state.notes
        if note.note_key
    }
    recorded_sources = db_port.resolve_note_sources(global_note_keys)
    ownership_conflicts = [
        f"{note_key}: {recorded_sources[note_key]} -> {source_id}"
        for note_key, source_id in source_by_note_key.items()
        if note_key in recorded_sources and recorded_sources[note_key] != source_id
    ]
    if ownership_conflicts:
        raise ValueError(
            "Cross-source note moves require an explicit sharing operation: "
            + "; ".join(sorted(ownership_conflicts))
        )
    pending_import_fingerprints: list[tuple[str, str, str]] = []

    with db_port.write_tx():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()
        initial_deck_names = set(deck_ids_by_name)
        deck_names_by_id = {
            deck_id: deck_name for deck_name, deck_id in deck_ids_by_name.items()
        }

        identity = resolve_import_note_identity(
            anki_port=anki_port,
            db_port=db_port,
            note_keys=global_note_keys,
            required_note_types=required_note_types,
        )
        anki_notes = identity.anki_notes
        note_ids_by_note_key = identity.note_ids_by_note_key
        pending_note_mappings = list(identity.pending_note_mappings)

        global_mapped_note_ids = {
            note_id
            for note_key, note_id in note_ids_by_note_key.items()
            if note_key in global_note_keys and note_id
        }

        # A single bulk cardsInfo fetch is faster than many smaller calls.
        all_card_ids = []
        for anki_note in anki_notes.values():
            all_card_ids.extend(anki_note.card_ids)
        anki_cards = anki_port.fetch_cards_info(all_card_ids)
        note_ids_by_deck_name = group_note_ids_by_deck_name(anki_cards)

        results = []
        pending_writes: list[PendingDeckWrite] = []
        markdown_deck_ids = set()
        markdown_deck_names = {
            file_stem_to_deck_name(deck_file.file_path.stem) for deck_file in deck_files
        }
        rename_candidates: set[tuple[str, str]] = set()

        for source_deck_file in source_deck_files:
            deck_file = source_deck_file.file_state
            deck_name = file_stem_to_deck_name(deck_file.file_path.stem)
            file_anki_note_ids = note_ids_by_deck_name.get(deck_name, set())

            sync_result, pending_write, moved_from_decks = sync_deck_to_anki(
                deck_file,
                note_types_by_name,
                anki_port,
                db_port,
                markdown_to_html,
                deck_names_by_id,
                deck_ids_by_name,
                anki_notes,
                anki_cards,
                note_ids_by_note_key,
                pending_note_mappings,
                note_import_fingerprints_by_note_key,
                pending_import_fingerprints,
                global_note_keys,
                global_mapped_note_ids,
                file_anki_note_ids,
            )

            # Track filename-based deck rename candidates, but defer applying
            # mapping cleanup until deck membership is refreshed.
            if (
                sync_result.name
                and sync_result.name not in initial_deck_names
                and len(moved_from_decks) == 1
            ):
                source_deck = next(iter(moved_from_decks))
                if (
                    source_deck != sync_result.name
                    and source_deck not in markdown_deck_names
                ):
                    rename_candidates.add((source_deck, sync_result.name))

            if sync_result.name and sync_result.name in deck_ids_by_name:
                markdown_deck_ids.add(deck_ids_by_name[sync_result.name])

            results.append(sync_result)
            pending_writes.append(pending_write)
            summary = sync_result.summary
            if summary.format() != "no changes":
                logger.debug(f"File '{deck_file.file_path.name}': {summary.format()}")

        if pending_note_mappings:
            db_port.upsert_note_links(pending_note_mappings)
        source_links: dict[str, list[tuple[str, int]]] = {}
        for source_deck_file in source_deck_files:
            source_id = source_deck_file.source.source_id
            for note in source_deck_file.file_state.notes:
                if not note.note_key:
                    continue
                note_id = note_ids_by_note_key.get(note.note_key)
                if note_id:
                    source_links.setdefault(source_id, []).append(
                        (note.note_key, note_id)
                    )
        for source_id, mappings in source_links.items():
            db_port.upsert_note_links(mappings, source_id=source_id)
        if pending_import_fingerprints:
            db_port.upsert_import_hashes(pending_import_fingerprints)

        flush_deck_metadata_writes(pending_writes)

        for source_deck_file in source_deck_files:
            deck_file = source_deck_file.file_state
            deck_name = file_stem_to_deck_name(deck_file.file_path.stem)
            deck_id = deck_ids_by_name.get(deck_name)
            if deck_id is not None:
                db_port.upsert_deck(
                    deck_name,
                    deck_id,
                    source_id=source_deck_file.source.source_id,
                    md_path=str(deck_file.file_path.relative_to(collection_dir)),
                )

        affected_note_ids = collect_membership_affected_note_ids(results)
        refresh_membership_for_affected_notes(
            affected_note_ids=affected_note_ids,
            anki_port=anki_port,
            note_ids_by_deck_name=note_ids_by_deck_name,
        )

        # Finalize deferred rename candidates only when the source deck is
        # fully drained after moves/deletes.
        for source_deck, target_deck in sorted(rename_candidates):
            if source_deck in note_ids_by_deck_name:
                continue
            db_port.delete_deck(source_deck)
            logger.info(
                f"Deck renamed from markdown file: '{source_deck}' -> '{target_deck}'"
            )

        untracked = []
        for deck_name, note_ids in note_ids_by_deck_name.items():
            deck_id = deck_ids_by_name.get(deck_name)
            if deck_id is None or deck_id in markdown_deck_ids:
                continue
            untracked.append(UntrackedDeck(deck_name, deck_id, list(note_ids)))

    protected_note_groups = [
        ProtectedNoteGroup(sync_result.name or "", sync_result.protected_keyless_notes)
        for sync_result in results
        if sync_result.protected_keyless_notes and sync_result.name
    ]

    return CollectionReport.for_import(
        results=results,
        untracked_decks=untracked,
        protected_note_groups=protected_note_groups,
    )

"""Collection-level Markdown-to-Anki sync."""

import logging
from pathlib import Path

from ankiops.anki import Anki
from ankiops.collection import file_stem_to_deck_name
from ankiops.deck_sources import DeckSource
from ankiops.interchange import ParsedDeck, ParsedSource
from ankiops.markdown import DeckFile
from ankiops.markdown_to_html import MarkdownToHTML
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


def _add_deleted_deck_files(
    collection_root: Path,
    state: SyncState,
    parsed_decks: list[tuple[DeckSource, ParsedDeck]],
    parsed_sources: tuple[ParsedSource, ...],
) -> set[Path]:
    source_by_path = {
        parsed_source.source.source_path: parsed_source.source
        for parsed_source in parsed_sources
    }
    existing_paths = {deck.path for _source, deck in parsed_decks}
    existing_deck_names = {deck.deck_name for _source, deck in parsed_decks}
    deleted_paths: set[Path] = set()
    for _deck_id, name, source_path, md_path in state.list_decks():
        source = source_by_path.get(source_path)
        if source is None or not md_path:
            continue
        path = collection_root / md_path
        if (
            path in existing_paths
            or path.exists()
            or not path.is_relative_to(source.root)
        ):
            continue
        if name in existing_deck_names:
            raise ValueError(
                f"Deck '{name}' is still present at another Markdown path."
            )
        parsed_decks.append(
            (
                source,
                ParsedDeck(
                    path=path,
                    deck_name=name,
                    parsed=DeckFile(file_path=path, raw_content="", notes=[]),
                ),
            )
        )
        existing_deck_names.add(name)
        deleted_paths.add(path)
    return deleted_paths


def sync_collection_to_anki(
    anki: Anki,
    state: SyncState,
    parsed_sources: tuple[ParsedSource, ...],
) -> CollectionReport:
    collection_root = parsed_sources[0].source.collection_root
    parsed_decks = [
        (parsed_source.source, deck)
        for parsed_source in parsed_sources
        for deck in parsed_source.decks
    ]
    note_types_by_name = {
        note_type.name: note_type
        for source in parsed_sources
        for note_type in source.note_types
    }
    deleted_deck_paths = _add_deleted_deck_files(
        collection_root,
        state,
        parsed_decks,
        parsed_sources,
    )

    deck_files = [deck.parsed for _source, deck in parsed_decks]
    global_note_keys = {
        note.note_key
        for deck_file in deck_files
        for note in deck_file.notes
        if note.note_key
    }
    required_note_types = sorted(note_types_by_name)
    markdown_to_html = MarkdownToHTML()

    note_import_fingerprints_by_note_key = state.resolve_import_hashes(global_note_keys)
    source_by_note_key = {
        note.note_key: source.source_path
        for source, deck in parsed_decks
        for note in deck.parsed.notes
        if note.note_key
    }
    recorded_sources = state.resolve_note_sources(global_note_keys)
    ownership_conflicts = [
        f"{note_key}: {recorded_sources[note_key]} -> {source_path}"
        for note_key, source_path in source_by_note_key.items()
        if note_key in recorded_sources and recorded_sources[note_key] != source_path
    ]
    if ownership_conflicts:
        raise ValueError(
            "Cross-source note moves require an explicit sharing operation: "
            + "; ".join(sorted(ownership_conflicts))
        )
    pending_import_fingerprints: list[tuple[str, str, str]] = []

    with state.write_tx():
        deck_ids_by_name = anki.fetch_deck_names_and_ids()
        initial_deck_names = set(deck_ids_by_name)
        deck_names_by_id = {
            deck_id: deck_name for deck_name, deck_id in deck_ids_by_name.items()
        }

        identity = resolve_import_note_identity(
            anki=anki,
            state=state,
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
        anki_cards = anki.fetch_cards_info(all_card_ids)
        note_ids_by_deck_name = group_note_ids_by_deck_name(anki_cards)

        results = []
        pending_writes: list[PendingDeckWrite] = []
        markdown_deck_ids = set()
        markdown_deck_names = {
            file_stem_to_deck_name(deck_file.file_path.stem)
            for deck_file in deck_files
            if deck_file.file_path not in deleted_deck_paths
        }
        rename_candidates: set[tuple[str, str]] = set()

        for _source, parsed_deck in parsed_decks:
            deck_file = parsed_deck.parsed
            deck_name = file_stem_to_deck_name(deck_file.file_path.stem)
            if (
                deck_file.file_path in deleted_deck_paths
                and deck_name not in initial_deck_names
            ):
                # A persisted deleted placeholder exists only to reconcile an
                # Anki deck that is still present. Never recreate one that is
                # already gone.
                continue
            file_anki_note_ids = note_ids_by_deck_name.get(deck_name, set())

            sync_result, pending_write, moved_from_decks = sync_deck_to_anki(
                deck_file,
                note_types_by_name,
                anki,
                state,
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

            if (
                deck_file.file_path not in deleted_deck_paths
                and sync_result.name
                and sync_result.name in deck_ids_by_name
            ):
                markdown_deck_ids.add(deck_ids_by_name[sync_result.name])

            results.append(sync_result)
            pending_writes.append(pending_write)
            summary = sync_result.summary
            if summary.format() != "no changes":
                logger.debug(f"File '{deck_file.file_path.name}': {summary.format()}")

        if pending_note_mappings:
            state.upsert_note_links(pending_note_mappings)
        source_links: dict[str, list[tuple[str, int]]] = {}
        for source, parsed_deck in parsed_decks:
            source_path = source.source_path
            for note in parsed_deck.parsed.notes:
                if not note.note_key:
                    continue
                note_id = note_ids_by_note_key.get(note.note_key)
                if note_id:
                    source_links.setdefault(source_path, []).append(
                        (note.note_key, note_id)
                    )
        for source_path, mappings in source_links.items():
            state.upsert_note_links(mappings, source_path=source_path)
        if pending_import_fingerprints:
            state.upsert_import_hashes(pending_import_fingerprints)

        flush_deck_metadata_writes(pending_writes)

        for source, parsed_deck in parsed_decks:
            deck_file = parsed_deck.parsed
            deck_name = file_stem_to_deck_name(deck_file.file_path.stem)
            if deck_file.file_path in deleted_deck_paths:
                state.delete_deck(deck_name)
                if not anki.fetch_card_ids_in_deck(deck_name):
                    anki.delete_empty_deck(deck_name)
                    logger.info(
                        f"Deck removed from Anki after Markdown deletion: '{deck_name}'"
                    )
                continue
            deck_id = deck_ids_by_name.get(deck_name)
            if deck_id is not None:
                state.upsert_deck(
                    deck_name,
                    deck_id,
                    source_path=source.source_path,
                    md_path=str(deck_file.file_path.relative_to(collection_root)),
                )

        affected_note_ids = collect_membership_affected_note_ids(results)
        refresh_membership_for_affected_notes(
            affected_note_ids=affected_note_ids,
            anki=anki,
            note_ids_by_deck_name=note_ids_by_deck_name,
        )

        # Finalize deferred rename candidates only when the source deck is
        # fully drained after moves/deletes.
        for source_deck, target_deck in sorted(rename_candidates):
            if source_deck in note_ids_by_deck_name:
                continue
            if anki.fetch_card_ids_in_deck(source_deck):
                continue
            anki.delete_empty_deck(source_deck)
            state.delete_deck(source_deck)
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

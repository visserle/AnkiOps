"""Use Case: Export Anki Notes to Markdown."""

import logging
from pathlib import Path

from ankiops.anki import Anki
from ankiops.collection import (
    deck_name_to_file_stem,
    file_stem_to_deck_name,
)
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_sources,
)
from ankiops.html_to_markdown import HTMLToMarkdown
from ankiops.markdown import (
    NOTE_SEPARATOR,
    DeckFile,
    format_note_type_comment,
    is_code_fence_line,
    is_note_type_comment,
    read_deck_file,
    render_notes_to_markdown,
    write_deck_file,
)
from ankiops.note_types import ANKIOPS_KEY_FIELD, NoteType
from ankiops.notes import (
    AnkiNote,
    Note,
    note_fingerprint,
)
from ankiops.sync.identity import assert_unique_export_note_keys
from ankiops.sync.report import (
    Change,
    ChangeType,
    CollectionReport,
    ProtectedNoteGroup,
    SyncReport,
)
from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)

_RESOLVED_NOTE_KEY = 0
_RESOLVED_NOTE_ID = 1
_RESOLVED_NOTE_VALUE = 2
_ResolvedDeckNote = tuple[str, int, Note]
_SYNC_ORDER = (
    ChangeType.DELETE,
    ChangeType.CREATE,
    ChangeType.CONVERT,
    ChangeType.UPDATE,
    ChangeType.SKIP,
    ChangeType.MOVE,
)


def _format_pending_note_type_conversion_error(
    *, note_key: str, markdown_note_type: str, anki_note_type: str
) -> str:
    return (
        f"Pending note type conversion for note_key {note_key}: Markdown uses "
        f"'{markdown_note_type}' but Anki has '{anki_note_type}'. Run "
        "'ankiops fa' before syncing with 'ankiops af'."
    )


def _from_html(
    anki_note: AnkiNote,
    config: NoteType,
    html_to_markdown: HTMLToMarkdown,
) -> Note:
    """Convert an Anki note into a Markdown-domain note."""
    fields = {}
    for field_config in config.fields:
        if field_config.name == ANKIOPS_KEY_FIELD.name:
            continue
        if field_config.name in anki_note.fields:
            md_val = html_to_markdown.convert(anki_note.fields[field_config.name])
            if md_val:
                fields[field_config.name] = md_val

    note_key = anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()
    return Note(
        note_key=note_key if note_key else None,
        note_type=anki_note.note_type,
        fields=fields,
        tags=anki_note.tags,
    )


def _load_deck_markdown_state(
    deck_name: str,
    existing_file_path: Path | None,
    collection_dir: Path,
    note_types: list[NoteType],
) -> tuple[Path, DeckFile, bool]:
    if existing_file_path and existing_file_path.exists():
        return (
            existing_file_path,
            read_deck_file(
                existing_file_path,
                note_types=note_types,
                context_root=collection_dir,
            ),
            False,
        )

    file_path = collection_dir / f"{deck_name_to_file_stem(deck_name)}.md"
    if file_path.exists():
        return (
            file_path,
            read_deck_file(
                file_path,
                note_types=note_types,
                context_root=collection_dir,
            ),
            False,
        )

    write_deck_file(file_path, "")
    return (
        file_path,
        read_deck_file(file_path, note_types=note_types, context_root=collection_dir),
        True,
    )


def _resolve_deck_notes(
    anki_notes: list[AnkiNote],
    config_by_name: dict[str, NoteType],
    html_to_markdown: HTMLToMarkdown,
    db_port: SyncState,
    result: SyncReport,
    note_keys_by_id: dict[int, str],
    pending_note_mappings: list[tuple[str, int]],
    note_export_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_export_fingerprints: list[tuple[str, str, str]],
    local_notes_by_note_key: dict[str, Note],
    local_notes_by_content: dict[str, Note],
) -> tuple[list[_ResolvedDeckNote], list[str]]:
    errors: list[str] = []
    resolved_notes: list[_ResolvedDeckNote] = []

    def _queue_note_mapping(note_key: str, note_id: int) -> None:
        if note_keys_by_id.get(note_id) == note_key:
            return
        note_keys_by_id[note_id] = note_key
        pending_note_mappings.append((note_key, note_id))

    def _queue_export_fingerprint(note_key: str, md_hash: str, anki_hash: str) -> None:
        if note_export_fingerprints_by_note_key.get(note_key) == (md_hash, anki_hash):
            return
        note_export_fingerprints_by_note_key[note_key] = (md_hash, anki_hash)
        pending_export_fingerprints.append((note_key, md_hash, anki_hash))

    for anki_note in anki_notes:
        note_key = note_keys_by_id.get(anki_note.note_id)
        embedded_note_key = anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()
        observed_anki_hash = note_fingerprint(
            anki_note.note_type,
            anki_note.fields,
            tags=anki_note.tags,
        )

        # Stale note_id->note_key mapping: prefer embedded note_key from Anki note.
        if note_key and embedded_note_key and note_key != embedded_note_key:
            _queue_note_mapping(embedded_note_key, anki_note.note_id)
            note_key = embedded_note_key
        elif not note_key and embedded_note_key:
            _queue_note_mapping(embedded_note_key, anki_note.note_id)
            note_key = embedded_note_key

        local_match = local_notes_by_note_key.get(note_key) if note_key else None
        if note_key and local_match:
            if local_match.note_type != anki_note.note_type:
                errors.append(
                    _format_pending_note_type_conversion_error(
                        note_key=note_key,
                        markdown_note_type=local_match.note_type,
                        anki_note_type=anki_note.note_type,
                    )
                )
                continue
            local_md_hash = note_fingerprint(
                local_match.note_type,
                local_match.fields,
                tags=local_match.tags,
            )
            cached = note_export_fingerprints_by_note_key.get(note_key)
            if cached == (local_md_hash, observed_anki_hash):
                result.add_change(
                    Change(ChangeType.SKIP, anki_note.note_id, local_match.identifier)
                )
                resolved_notes.append((note_key, anki_note.note_id, local_match))
                _queue_export_fingerprint(note_key, local_md_hash, observed_anki_hash)
                continue

        cfg = config_by_name.get(anki_note.note_type)
        if not cfg:
            errors.append(
                f"Unknown note type {anki_note.note_type} for note {anki_note.note_id}"
            )
            continue

        domain_note = _from_html(anki_note, cfg, html_to_markdown)

        if not note_key:
            # Match by content when no mapped/embedded note_key exists.
            first_line = domain_note.first_field_line()
            match = local_notes_by_content.get(first_line)
            if match and match.note_key:
                note_key = match.note_key
                _queue_note_mapping(note_key, anki_note.note_id)
            else:
                note_key = db_port.generate_note_key()
                _queue_note_mapping(note_key, anki_note.note_id)

        domain_note.note_key = note_key
        local_match = local_notes_by_note_key.get(note_key)
        if (
            local_match
            and local_match.fields == domain_note.fields
            and local_match.tags == domain_note.tags
        ):
            result.add_change(
                Change(ChangeType.SKIP, anki_note.note_id, domain_note.identifier)
            )
            resolved_notes.append((note_key, anki_note.note_id, local_match))
            md_hash = note_fingerprint(
                local_match.note_type,
                local_match.fields,
                tags=local_match.tags,
            )
            _queue_export_fingerprint(note_key, md_hash, observed_anki_hash)
            continue

        if not local_match:
            result.add_change(
                Change(
                    ChangeType.CREATE,
                    anki_note.note_id,
                    domain_note.identifier,
                )
            )
            logger.debug(
                "  Create markdown note from Anki note_id=%s (note_key=%s)",
                anki_note.note_id,
                note_key,
            )
        else:
            result.add_change(
                Change(
                    ChangeType.UPDATE,
                    anki_note.note_id,
                    domain_note.identifier,
                )
            )
            logger.debug(
                "  Update markdown note from Anki note_id=%s (note_key=%s)",
                anki_note.note_id,
                note_key,
            )

        resolved_notes.append((note_key, anki_note.note_id, domain_note))
        md_hash = note_fingerprint(
            domain_note.note_type,
            domain_note.fields,
            tags=domain_note.tags,
        )
        _queue_export_fingerprint(note_key, md_hash, observed_anki_hash)

    return resolved_notes, errors


def _order_resolved_notes(
    resolved_notes: list[_ResolvedDeckNote],
    existing_notes: list[Note],
    is_first_export: bool,
) -> tuple[list[Note], int]:
    if is_first_export:
        return [
            resolved[_RESOLVED_NOTE_VALUE]
            for resolved in sorted(
                resolved_notes,
                key=lambda resolved_note: resolved_note[_RESOLVED_NOTE_ID],
            )
        ], 0

    resolved_by_note_key = {
        resolved[_RESOLVED_NOTE_KEY]: resolved for resolved in resolved_notes
    }
    resolved_by_fingerprint: dict[str, list[_ResolvedDeckNote]] = {}
    for resolved in resolved_notes:
        resolved_note = resolved[_RESOLVED_NOTE_VALUE]
        resolved_hash = note_fingerprint(
            resolved_note.note_type,
            resolved_note.fields,
            tags=resolved_note.tags,
        )
        resolved_by_fingerprint.setdefault(resolved_hash, []).append(resolved)

    consumed_note_keys: set[str] = set()
    ordered_notes: list[Note] = []
    protected_keyless_count = 0

    for existing_note in existing_notes:
        if existing_note.note_key:
            resolved = resolved_by_note_key.get(existing_note.note_key)
            if not resolved:
                continue
            resolved_key = resolved[_RESOLVED_NOTE_KEY]
            if resolved_key in consumed_note_keys:
                continue
            ordered_notes.append(resolved[_RESOLVED_NOTE_VALUE])
            consumed_note_keys.add(resolved_key)
            continue

        existing_hash = note_fingerprint(
            existing_note.note_type,
            existing_note.fields,
            tags=existing_note.tags,
        )
        candidates = resolved_by_fingerprint.get(existing_hash, [])
        matched = next(
            (
                candidate
                for candidate in candidates
                if candidate[_RESOLVED_NOTE_KEY] not in consumed_note_keys
            ),
            None,
        )
        if matched is not None:
            ordered_notes.append(matched[_RESOLVED_NOTE_VALUE])
            consumed_note_keys.add(matched[_RESOLVED_NOTE_KEY])
            continue

        ordered_notes.append(existing_note)
        protected_keyless_count += 1

    remaining = sorted(
        [
            resolved
            for resolved in resolved_notes
            if resolved[_RESOLVED_NOTE_KEY] not in consumed_note_keys
        ],
        key=lambda resolved: resolved[_RESOLVED_NOTE_ID],
    )
    ordered_notes.extend(resolved[_RESOLVED_NOTE_VALUE] for resolved in remaining)
    return ordered_notes, protected_keyless_count


def _can_skip_markdown_rebuild(
    *,
    is_first_export: bool,
    fs: DeckFile,
    final_notes: list[Note],
    result: SyncReport,
    resolve_errors: list[str],
) -> bool:
    if is_first_export or resolve_errors:
        return False
    if result.has_changes(ChangeType.CREATE, ChangeType.UPDATE):
        return False
    if len(final_notes) != len(fs.notes):
        return False

    for existing_note, final_note in zip(fs.notes, final_notes):
        if final_note is not existing_note:
            return False
    return _has_current_note_type_metadata(fs, final_notes)


def _has_current_note_type_metadata(fs: DeckFile, notes: list[Note]) -> bool:
    blocks = fs.raw_content.split(NOTE_SEPARATOR)
    note_idx = 0
    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block or not stripped_block.replace("-", ""):
            continue
        if note_idx >= len(notes):
            return False
        note = notes[note_idx]
        note_idx += 1
        if not _block_has_current_note_type_metadata(block, note.note_type):
            return False
    return note_idx == len(notes)


def _block_has_current_note_type_metadata(block: str, note_type: str) -> bool:
    expected = format_note_type_comment(note_type)
    found = False
    in_code_block = False
    for line in block.split("\n"):
        stripped = line.lstrip()
        if is_code_fence_line(stripped):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not is_note_type_comment(line):
            continue
        if line.strip() != expected:
            return False
        found = True
    return found


def _sync_deck(
    deck_name: str,
    anki_notes: list[AnkiNote],
    config_by_name: dict[str, NoteType],
    existing_file_path: Path | None,
    collection_dir: Path,
    html_to_markdown: HTMLToMarkdown,
    db_port: SyncState,
    note_keys_by_id: dict[int, str],
    pending_note_mappings: list[tuple[str, int]],
    note_export_fingerprints_by_note_key: dict[str, tuple[str, str]],
    pending_export_fingerprints: list[tuple[str, str, str]],
) -> SyncReport:
    result = SyncReport.for_notes(name=deck_name, file_path=existing_file_path)
    file_path, fs, is_first_export = _load_deck_markdown_state(
        deck_name=deck_name,
        existing_file_path=existing_file_path,
        collection_dir=collection_dir,
        note_types=list(config_by_name.values()),
    )
    result.file_path = file_path

    local_notes_by_note_key = {
        local_note.note_key: local_note
        for local_note in fs.notes
        if local_note.note_key
    }
    local_notes_by_content = {
        local_note.first_field_line(): local_note
        for local_note in fs.notes
        if not local_note.note_key
    }

    resolved_notes, resolve_errors = _resolve_deck_notes(
        anki_notes=anki_notes,
        config_by_name=config_by_name,
        html_to_markdown=html_to_markdown,
        db_port=db_port,
        result=result,
        note_keys_by_id=note_keys_by_id,
        pending_note_mappings=pending_note_mappings,
        note_export_fingerprints_by_note_key=note_export_fingerprints_by_note_key,
        pending_export_fingerprints=pending_export_fingerprints,
        local_notes_by_note_key=local_notes_by_note_key,
        local_notes_by_content=local_notes_by_content,
    )
    result.errors.extend(resolve_errors)
    if resolve_errors:
        result.order_changes(order=_SYNC_ORDER)
        return result

    final_notes, protected_keyless_count = _order_resolved_notes(
        resolved_notes=resolved_notes,
        existing_notes=fs.notes,
        is_first_export=is_first_export,
    )
    result.protected_keyless_notes = protected_keyless_count

    if _can_skip_markdown_rebuild(
        is_first_export=is_first_export,
        fs=fs,
        final_notes=final_notes,
        result=result,
        resolve_errors=resolve_errors,
    ):
        result.order_changes(order=_SYNC_ORDER)
        return result

    final_text = render_notes_to_markdown(
        notes=final_notes,
        note_types_by_name=config_by_name,
    )

    if final_text.strip() != fs.raw_content.strip():
        write_deck_file(result.file_path, final_text)

    # Detect deleted notes
    final_note_keys = {
        final_note.note_key for final_note in final_notes if final_note.note_key
    }
    for old_note in fs.notes:
        if old_note.note_key and old_note.note_key not in final_note_keys:
            # Note was deleted in Anki
            db_port.delete_note_links_by_keys([old_note.note_key])
            result.add_change(Change(ChangeType.DELETE, None, old_note.identifier))
            logger.debug(
                "  Delete markdown note for removed Anki note (note_key=%s)",
                old_note.note_key,
            )

    result.order_changes(order=_SYNC_ORDER)
    return result


def _split_orphan_file_notes(
    *,
    md_file: Path,
    collection_dir: Path,
    note_types: list[NoteType],
) -> tuple[list[Note], list[Note]] | None:
    try:
        file_state = read_deck_file(
            md_file,
            note_types=note_types,
            context_root=collection_dir,
        )
    except Exception as error:  # pragma: no cover - defensive fail-safe
        logger.warning(
            f"Preserving orphan markdown file '{md_file.name}' because it "
            f"could not be parsed during export cleanup: {error}"
        )
        return None

    keyless_notes: list[Note] = []
    keyed_notes: list[Note] = []
    for note in file_state.notes:
        if note.note_key:
            keyed_notes.append(note)
        else:
            keyless_notes.append(note)
    return keyless_notes, keyed_notes


def sync_collection_from_anki(
    anki_port: Anki,
    db_port: SyncState,
    collection_dir: Path,
    note_types_dir: Path,
) -> CollectionReport:
    sources = discover_deck_sources(collection_dir, note_types_dir=note_types_dir)
    local_source = sources[0]
    source_configs = load_note_types_for_sources(sources)
    configs = [
        config
        for source_config in source_configs
        for config in source_config.note_types
    ]
    config_by_name = {config.name: config for config in configs}
    html_to_markdown = HTMLToMarkdown()
    with db_port.write_tx():
        deck_ids_by_name = anki_port.fetch_deck_names_and_ids()

        all_note_ids = anki_port.fetch_all_note_ids([config.name for config in configs])
        anki_notes = anki_port.fetch_notes_info(all_note_ids)
        note_keys_by_id = db_port.resolve_note_keys(all_note_ids)
        assert_unique_export_note_keys(
            anki_notes=anki_notes,
            note_keys_by_id=note_keys_by_id,
        )
        pending_note_mappings: list[tuple[str, int]] = []
        note_key_candidates = set(note_keys_by_id.values())
        for anki_note in anki_notes.values():
            embedded_note_key = anki_note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()
            if embedded_note_key:
                note_key_candidates.add(embedded_note_key)
        note_export_fingerprints_by_note_key = db_port.resolve_export_hashes(
            note_key_candidates
        )
        pending_export_fingerprints: list[tuple[str, str, str]] = []

        # Fetch cards info to map notes to decks
        all_card_ids = []
        for anki_note in anki_notes.values():
            all_card_ids.extend(anki_note.card_ids)
        anki_cards = anki_port.fetch_cards_info(all_card_ids)

        notes_by_deck = {}
        for note_id, anki_note in anki_notes.items():
            if not anki_note.card_ids:
                continue
            primary_card_info = anki_cards.get(anki_note.card_ids[0])
            if not primary_card_info:
                continue
            deck_name = primary_card_info.get("deckName")
            if deck_name is None:
                continue
            notes_by_deck.setdefault(deck_name, []).append(anki_note)

        md_files: list[Path] = []
        source_by_file: dict[Path, DeckSource] = {}
        file_map_by_deck_name: dict[str, tuple[DeckSource, Path]] = {}
        deck_conflicts: list[str] = []
        for source in sources:
            for md_file in source.deck_files():
                md_files.append(md_file)
                source_by_file[md_file] = source
                deck_name = file_stem_to_deck_name(md_file.stem)
                previous = file_map_by_deck_name.get(deck_name)
                if previous is not None:
                    deck_conflicts.append(
                        f"Deck '{deck_name}' is defined by both "
                        f"{previous[0].display_name}:{previous[1].name} and "
                        f"{source.display_name}:{md_file.name}"
                    )
                    continue
                file_map_by_deck_name[deck_name] = (source, md_file)
        if deck_conflicts:
            raise ValueError(
                "Aborting export: deck ownership conflicts found: "
                + "; ".join(deck_conflicts)
            )

        results = []
        protected_note_groups: list[ProtectedNoteGroup] = []

        for deck_name, notes in notes_by_deck.items():
            if deck_name == "Default":
                continue

            deck_id = deck_ids_by_name[deck_name]

            # Rename detection: does this deck_id exist under a different name?
            old_name = db_port.resolve_deck_name(deck_id)
            safe_name = deck_name_to_file_stem(deck_name)
            if old_name and old_name != deck_name:
                old_entry = file_map_by_deck_name.get(old_name)
                if old_entry is not None:
                    old_source, old_path = old_entry
                    new_path = old_path.parent / f"{safe_name}.md"
                    old_path.rename(new_path)
                    source_by_file.pop(old_path, None)
                    source_by_file[new_path] = old_source
                    file_map_by_deck_name[deck_name] = (old_source, new_path)
                    del file_map_by_deck_name[old_name]
                    logger.info(f"Deck renamed: '{old_name}' → '{deck_name}'")

            target_source, target_file = file_map_by_deck_name.get(
                deck_name,
                (local_source, None),
            )

            sync_result = _sync_deck(
                deck_name,
                notes,
                config_by_name,
                target_file,
                target_source.root,
                html_to_markdown,
                db_port,
                note_keys_by_id,
                pending_note_mappings,
                note_export_fingerprints_by_note_key,
                pending_export_fingerprints,
            )
            db_port.upsert_deck(deck_name, deck_id)
            results.append(sync_result)
            if sync_result.protected_keyless_notes:
                protected_note_groups.append(
                    ProtectedNoteGroup(deck_name, sync_result.protected_keyless_notes)
                )
            summary = sync_result.summary
            if summary.format() != "no changes":
                logger.debug(f"Deck '{deck_name}': {summary.format()}")

        if pending_note_mappings:
            db_port.upsert_note_links(pending_note_mappings)
        if pending_export_fingerprints:
            db_port.upsert_export_hashes(pending_export_fingerprints)

        extra_changes = []
        active_files = {
            sync_result.file_path for sync_result in results if sync_result.file_path
        }
        active_note_keys = {
            note_key for note_key in note_keys_by_id.values() if note_key
        }
        for md_file in md_files:
            # A prior deck-rename step can move this file already.
            if not md_file.exists():
                continue
            if md_file not in active_files:
                deck_name = file_stem_to_deck_name(md_file.stem)
                source = source_by_file.get(md_file, local_source)
                orphan_split = _split_orphan_file_notes(
                    md_file=md_file,
                    collection_dir=source.root,
                    note_types=configs,
                )
                if orphan_split is None:
                    db_port.delete_deck(deck_name)
                    continue
                keyless_notes, keyed_notes = orphan_split
                stale_orphan_note_keys = sorted(
                    {
                        note.note_key
                        for note in keyed_notes
                        if note.note_key and note.note_key not in active_note_keys
                    }
                )
                if stale_orphan_note_keys:
                    db_port.delete_note_links_by_keys(stale_orphan_note_keys)
                protected_note_count = len(keyless_notes)

                if protected_note_count:
                    db_port.delete_deck(deck_name)
                    protected_note_groups.append(
                        ProtectedNoteGroup(deck_name, protected_note_count)
                    )
                    if keyed_notes:
                        final_text = render_notes_to_markdown(
                            notes=keyless_notes,
                            note_types_by_name=config_by_name,
                        )
                        write_deck_file(md_file, final_text)
                        extra_changes.extend(
                            Change(ChangeType.DELETE, None, orphan.identifier)
                            for orphan in keyed_notes
                        )
                    continue

                db_port.delete_deck(deck_name)
                md_file.unlink()
                if not keyed_notes:
                    logger.info(
                        "Removed empty orphan markdown deck file: %s",
                        md_file.name,
                    )
                extra_changes.extend(
                    Change(ChangeType.DELETE, None, orphan.identifier)
                    for orphan in keyed_notes
                )
        return CollectionReport.for_export(
            results=results,
            extra_changes=extra_changes,
            protected_note_groups=protected_note_groups,
        )

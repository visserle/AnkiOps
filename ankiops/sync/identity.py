"""Resolve managed note identity before sync planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ankiops.note_types import ANKIOPS_KEY_FIELD
from ankiops.notes import AnkiNote


class AnkiIdentityReader(Protocol):
    def fetch_all_note_ids(self, required_types: list[str]) -> list[int]: ...

    def fetch_note_ids_by_note_keys(
        self,
        note_keys: set[str],
    ) -> dict[str, list[int]]: ...

    def fetch_notes_info(self, note_ids: list[int]) -> dict[int, AnkiNote]: ...


class NoteIdentityState(Protocol):
    def resolve_note_ids(self, note_keys: set[str]) -> dict[str, int]: ...


@dataclass(frozen=True)
class ImportNoteIdentity:
    anki_notes: dict[int, AnkiNote]
    note_ids_by_note_key: dict[str, int]
    pending_note_mappings: list[tuple[str, int]]


def _embedded_note_key(note: AnkiNote) -> str:
    return note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()


def _merge_notes(
    target: dict[int, AnkiNote],
    new_notes: dict[int, AnkiNote],
) -> None:
    for note_id, note in new_notes.items():
        target[note_id] = note


def _embedded_note_ids_by_key(
    anki_notes: dict[int, AnkiNote],
    note_keys: set[str],
) -> dict[str, set[int]]:
    note_ids_by_key: dict[str, set[int]] = {}
    for note in anki_notes.values():
        embedded_note_key = _embedded_note_key(note)
        if embedded_note_key and embedded_note_key in note_keys:
            note_ids_by_key.setdefault(embedded_note_key, set()).add(note.note_id)
    return note_ids_by_key


def _candidate_note_ids(
    *,
    note_key: str,
    mapped_note_id: int | None,
    anki_notes: dict[int, AnkiNote],
    embedded_note_ids_by_key: dict[str, set[int]],
) -> set[int]:
    candidates = set(embedded_note_ids_by_key.get(note_key, set()))
    if mapped_note_id is None:
        return candidates

    mapped_note = anki_notes.get(mapped_note_id)
    if mapped_note is None:
        return candidates

    embedded_note_key = _embedded_note_key(mapped_note)
    if not embedded_note_key or embedded_note_key == note_key:
        candidates.add(mapped_note_id)
    return candidates


def _needs_key_field_lookup(
    *,
    note_key: str,
    mapped_note_id: int | None,
    anki_notes: dict[int, AnkiNote],
    candidates: set[int],
) -> bool:
    if not candidates:
        return True
    if mapped_note_id is None:
        return False
    mapped_note = anki_notes.get(mapped_note_id)
    if mapped_note is None:
        return True
    return not _embedded_note_key(mapped_note)


def _duplicate_errors(
    *,
    note_keys: set[str],
    db_mappings: dict[str, int],
    anki_notes: dict[int, AnkiNote],
    embedded_note_ids_by_key: dict[str, set[int]],
) -> list[str]:
    note_ids_by_key: dict[str, set[int]] = {}
    for note_key in sorted(note_keys):
        candidates = _candidate_note_ids(
            note_key=note_key,
            mapped_note_id=db_mappings.get(note_key),
            anki_notes=anki_notes,
            embedded_note_ids_by_key=embedded_note_ids_by_key,
        )
        note_ids_by_key[note_key] = candidates
    return _duplicate_key_errors(note_ids_by_key)


def _duplicate_key_errors(note_ids_by_key: dict[str, set[int]]) -> list[str]:
    return [
        f"Duplicate AnkiOps Key '{note_key}' found on Anki note IDs {sorted(note_ids)}"
        for note_key, note_ids in sorted(note_ids_by_key.items())
        if len(note_ids) > 1
    ]


def _resolve_note_ids_by_key(
    *,
    note_keys: set[str],
    db_mappings: dict[str, int],
    anki_notes: dict[int, AnkiNote],
    embedded_note_ids_by_key: dict[str, set[int]],
) -> dict[str, int]:
    note_ids_by_key: dict[str, int] = {}
    for note_key in sorted(note_keys):
        candidates = _candidate_note_ids(
            note_key=note_key,
            mapped_note_id=db_mappings.get(note_key),
            anki_notes=anki_notes,
            embedded_note_ids_by_key=embedded_note_ids_by_key,
        )
        if candidates:
            note_ids_by_key[note_key] = min(candidates)
    return note_ids_by_key


def resolve_import_note_identity(
    *,
    anki: AnkiIdentityReader,
    state: NoteIdentityState,
    note_keys: set[str],
    required_note_types: list[str],
) -> ImportNoteIdentity:
    """Resolve keyed Markdown notes to Anki notes before creating anything."""
    if not note_keys:
        all_note_ids = set(anki.fetch_all_note_ids(required_note_types))
        anki_notes = anki.fetch_notes_info(sorted(all_note_ids))
        return ImportNoteIdentity(
            anki_notes=anki_notes,
            note_ids_by_note_key={},
            pending_note_mappings=[],
        )

    db_mappings = state.resolve_note_ids(note_keys)
    all_note_ids = set(anki.fetch_all_note_ids(required_note_types))
    all_note_ids.update(db_mappings.values())
    anki_notes = anki.fetch_notes_info(sorted(all_note_ids))

    embedded_note_ids = _embedded_note_ids_by_key(anki_notes, note_keys)
    key_field_lookup_note_keys = {
        note_key
        for note_key in note_keys
        if _needs_key_field_lookup(
            note_key=note_key,
            mapped_note_id=db_mappings.get(note_key),
            anki_notes=anki_notes,
            candidates=_candidate_note_ids(
                note_key=note_key,
                mapped_note_id=db_mappings.get(note_key),
                anki_notes=anki_notes,
                embedded_note_ids_by_key=embedded_note_ids,
            ),
        )
    }

    key_field_note_ids_by_key = (
        anki.fetch_note_ids_by_note_keys(key_field_lookup_note_keys)
        if key_field_lookup_note_keys
        else {}
    )
    key_field_note_ids = {
        note_id
        for note_ids in key_field_note_ids_by_key.values()
        for note_id in note_ids
    }
    missing_key_field_note_ids = key_field_note_ids - set(anki_notes)
    if missing_key_field_note_ids:
        _merge_notes(
            anki_notes,
            anki.fetch_notes_info(sorted(missing_key_field_note_ids)),
        )

    embedded_note_ids = _embedded_note_ids_by_key(anki_notes, note_keys)
    duplicate_errors = _duplicate_errors(
        note_keys=note_keys,
        db_mappings=db_mappings,
        anki_notes=anki_notes,
        embedded_note_ids_by_key=embedded_note_ids,
    )
    if duplicate_errors:
        raise ValueError(
            "Aborting import: duplicate AnkiOps Key values found in Anki: "
            + "; ".join(duplicate_errors)
        )

    note_ids_by_key = _resolve_note_ids_by_key(
        note_keys=note_keys,
        db_mappings=db_mappings,
        anki_notes=anki_notes,
        embedded_note_ids_by_key=embedded_note_ids,
    )
    pending_note_mappings = [
        (note_key, note_id)
        for note_key, note_id in sorted(note_ids_by_key.items())
        if db_mappings.get(note_key) != note_id
    ]

    return ImportNoteIdentity(
        anki_notes=anki_notes,
        note_ids_by_note_key=note_ids_by_key,
        pending_note_mappings=pending_note_mappings,
    )


def assert_unique_export_note_keys(
    *,
    anki_notes: dict[int, AnkiNote],
    note_keys_by_id: dict[int, str],
) -> None:
    """Block export when two Anki notes claim the same managed identity."""
    note_ids_by_key: dict[str, set[int]] = {}
    for anki_note in anki_notes.values():
        note_key = _embedded_note_key(anki_note) or note_keys_by_id.get(
            anki_note.note_id,
            "",
        )
        if not note_key:
            continue
        note_ids_by_key.setdefault(note_key, set()).add(anki_note.note_id)

    duplicate_errors = _duplicate_key_errors(note_ids_by_key)
    if duplicate_errors:
        raise ValueError(
            "Aborting export: duplicate AnkiOps Key values found in Anki: "
            + "; ".join(duplicate_errors)
        )

from __future__ import annotations

import pytest

from ankiops.notes import AnkiNote
from ankiops.sync.identity import (
    assert_unique_export_note_keys,
    resolve_import_note_identity,
)


class FakeAnkiIdentity:
    def __init__(
        self,
        notes: dict[int, AnkiNote],
        *,
        model_note_ids: list[int] | None = None,
        note_ids_by_key: dict[str, list[int]] | None = None,
    ):
        self.notes = notes
        self.model_note_ids = model_note_ids or []
        self.note_ids_by_key = note_ids_by_key or {}
        self.key_field_lookup_calls: list[set[str]] = []

    def fetch_all_note_ids(self, required_types: list[str]) -> list[int]:
        return self.model_note_ids

    def fetch_note_ids_by_note_keys(
        self,
        note_keys: set[str],
    ) -> dict[str, list[int]]:
        self.key_field_lookup_calls.append(set(note_keys))
        return {
            note_key: self.note_ids_by_key.get(note_key, []) for note_key in note_keys
        }

    def fetch_notes_info(self, note_ids: list[int]) -> dict[int, AnkiNote]:
        return {
            note_id: self.notes[note_id]
            for note_id in note_ids
            if note_id in self.notes
        }


class FakeDbIdentity:
    def __init__(self, mappings: dict[str, int]):
        self.mappings = mappings

    def resolve_note_ids(self, note_keys: set[str]) -> dict[str, int]:
        return {
            note_key: note_id
            for note_key, note_id in self.mappings.items()
            if note_key in note_keys
        }


def _note(
    note_id: int,
    note_type: str = "AnkiOpsQA",
    note_key: str = "key-1",
) -> AnkiNote:
    return AnkiNote(
        note_id=note_id,
        note_type=note_type,
        fields={"AnkiOps Key": note_key, "Question": "Q", "Answer": "A"},
        card_ids=[note_id + 1000],
    )


def test_resolves_healthy_db_mapping_without_key_field_lookup():
    anki = FakeAnkiIdentity(
        {101: _note(101)},
        model_note_ids=[101],
    )
    db = FakeDbIdentity({"key-1": 101})

    identity = resolve_import_note_identity(
        anki_port=anki,
        db_port=db,
        note_keys={"key-1"},
        required_note_types=["AnkiOpsQA"],
    )

    assert identity.note_ids_by_note_key == {"key-1": 101}
    assert identity.pending_note_mappings == []
    assert anki.key_field_lookup_calls == []


def test_indexes_missing_db_mapping_from_configured_model_snapshot():
    anki = FakeAnkiIdentity(
        {101: _note(101)},
        model_note_ids=[101],
    )
    db = FakeDbIdentity({})

    identity = resolve_import_note_identity(
        anki_port=anki,
        db_port=db,
        note_keys={"key-1"},
        required_note_types=["AnkiOpsQA"],
    )

    assert identity.note_ids_by_note_key == {"key-1": 101}
    assert identity.pending_note_mappings == [("key-1", 101)]
    assert anki.key_field_lookup_calls == []


def test_uses_db_mapped_note_even_when_old_model_is_not_configured():
    anki = FakeAnkiIdentity(
        {101: _note(101, note_type="OldQA")},
        model_note_ids=[],
    )
    db = FakeDbIdentity({"key-1": 101})

    identity = resolve_import_note_identity(
        anki_port=anki,
        db_port=db,
        note_keys={"key-1"},
        required_note_types=["NewQA"],
    )

    assert identity.note_ids_by_note_key == {"key-1": 101}
    assert identity.anki_notes[101].note_type == "OldQA"
    assert anki.key_field_lookup_calls == []


def test_uses_key_field_lookup_only_for_unresolved_identity():
    anki = FakeAnkiIdentity(
        {101: _note(101, note_type="OldQA")},
        model_note_ids=[],
        note_ids_by_key={"key-1": [101]},
    )
    db = FakeDbIdentity({})

    identity = resolve_import_note_identity(
        anki_port=anki,
        db_port=db,
        note_keys={"key-1"},
        required_note_types=["NewQA"],
    )

    assert identity.note_ids_by_note_key == {"key-1": 101}
    assert identity.pending_note_mappings == [("key-1", 101)]
    assert anki.key_field_lookup_calls == [{"key-1"}]


def test_blocks_when_key_field_lookup_would_duplicate_existing_key():
    anki = FakeAnkiIdentity(
        {
            101: _note(101, note_key=""),
            202: _note(202, note_key="key-1"),
        },
        model_note_ids=[101],
        note_ids_by_key={"key-1": [202]},
    )
    db = FakeDbIdentity({"key-1": 101})

    with pytest.raises(ValueError, match="Duplicate AnkiOps Key 'key-1'"):
        resolve_import_note_identity(
            anki_port=anki,
            db_port=db,
            note_keys={"key-1"},
            required_note_types=["AnkiOpsQA"],
        )

    assert anki.key_field_lookup_calls == [{"key-1"}]


def test_export_identity_blocks_duplicate_embedded_keys():
    notes = {
        101: _note(101, note_key="key-1"),
        202: _note(202, note_key="key-1"),
    }

    with pytest.raises(ValueError, match="Aborting export"):
        assert_unique_export_note_keys(
            anki_notes=notes,
            note_keys_by_id={},
        )


def test_export_identity_blocks_db_mapping_and_embedded_key_collision():
    notes = {
        101: _note(101, note_key=""),
        202: _note(202, note_key="key-1"),
    }

    with pytest.raises(ValueError, match="Duplicate AnkiOps Key 'key-1'"):
        assert_unique_export_note_keys(
            anki_notes=notes,
            note_keys_by_id={101: "key-1"},
        )

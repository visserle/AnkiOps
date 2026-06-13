from __future__ import annotations

import pytest

from anki_addon.actions import AnkiOpsConnectActionError
from anki_addon.note_type_conversion import convert_notes_to_note_type
from tests.unit.addon.fakes import _FakeCollection, _model


def test_convert_notes_to_note_type_converts_selected_notes_only():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")
    col.add_note(102, col.old_model, "key-102")

    result = convert_notes_to_note_type(
        col,
        [101],
        "AnkiOpsQA",
        "shared/owner/repo/AnkiOpsQA",
    )

    assert result == {"changed": 1}
    assert col.notes[101].note_type()["name"] == "shared/owner/repo/AnkiOpsQA"
    assert col.notes[102].note_type()["name"] == "AnkiOpsQA"


def test_convert_notes_to_note_type_blocks_missing_model():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="New note type not found"):
        convert_notes_to_note_type(col, [101], "AnkiOpsQA", "missing")


def test_convert_notes_to_note_type_blocks_wrong_current_model():
    col = _FakeCollection()
    col.add_note(101, col.new_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="expected 'AnkiOpsQA'"):
        convert_notes_to_note_type(
            col,
            [101],
            "AnkiOpsQA",
            "shared/owner/repo/AnkiOpsQA",
        )


def test_convert_notes_to_note_type_blocks_missing_ankiops_key_field():
    col = _FakeCollection()
    col.old_model = _model(1, "AnkiOpsQA", fields=["Question", "Answer"])
    col.models_by_name["AnkiOpsQA"] = col.old_model
    col.models_by_id[1] = col.old_model
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="lacks AnkiOps Key"):
        convert_notes_to_note_type(
            col,
            [101],
            "AnkiOpsQA",
            "shared/owner/repo/AnkiOpsQA",
        )


def test_convert_notes_to_note_type_blocks_mismatched_fields():
    col = _FakeCollection()
    col.new_model = _model(
        2,
        "shared/owner/repo/AnkiOpsQA",
        fields=["Question", "Different", "AnkiOps Key"],
    )
    col.models_by_name[col.new_model["name"]] = col.new_model
    col.models_by_id[2] = col.new_model
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="field names differ"):
        convert_notes_to_note_type(
            col,
            [101],
            "AnkiOpsQA",
            "shared/owner/repo/AnkiOpsQA",
        )


def test_convert_notes_to_note_type_blocks_post_verification_failures():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")
    col.mutate_card_due_on_change = True

    with pytest.raises(
        AnkiOpsConnectActionError,
        match="Post-conversion verification failed",
    ):
        convert_notes_to_note_type(
            col,
            [101],
            "AnkiOpsQA",
            "shared/owner/repo/AnkiOpsQA",
        )

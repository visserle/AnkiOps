"""Error-path tests for AnkiAdapter.apply_note_changes."""

from __future__ import annotations

from unittest.mock import patch

from ankiops.anki import AnkiAdapter
from ankiops.anki_client import AnkiConnectError
from ankiops.models import Change, ChangeType, Note


def _make_update_change() -> Change:
    return Change(
        ChangeType.UPDATE,
        101,
        "note_key: update",
        {"html_fields": {"Question": "Q", "Answer": "A2"}},
    )


def _make_delete_change() -> Change:
    return Change(ChangeType.DELETE, 202, "note_key: delete")


def _make_create_change() -> Change:
    return Change(
        ChangeType.CREATE,
        None,
        "note_key: create",
        {
            "note": Note(note_key=None, note_type="AnkiOpsQA", fields={}),
            "html_fields": {"Question": "Q", "Answer": "A"},
            "note_key": "created-key",
        },
    )


def test_apply_note_changes_collects_partial_errors():
    adapter = AnkiAdapter()
    fake_non_create_result = [
        None,
        "change deck failed",
        "update failed",
        "delete failed",
    ]

    with patch(
        "ankiops.anki.invoke",
        side_effect=[fake_non_create_result, ["create failed"]],
    ):
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckX",
            needs_create_deck=True,
            creates=[_make_create_change()],
            updates=[_make_update_change()],
            deletes=[_make_delete_change()],
            cards_to_move=[1001],
        )

    assert created_ids == []
    assert len(errors) == 4
    assert any("Update failed (note_key: update)" in error for error in errors)
    assert any("Create failed (note_key: create)" in error for error in errors)
    assert any("Failed change_deck" in error for error in errors)
    assert any("Failed delete" in error for error in errors)


def test_apply_note_changes_surfaces_ankiconnect_exception():
    adapter = AnkiAdapter()

    with patch("ankiops.anki.invoke", side_effect=AnkiConnectError("boom")):
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckY",
            needs_create_deck=False,
            creates=[],
            updates=[],
            deletes=[],
            cards_to_move=[1002],
        )

    assert created_ids == []
    assert errors == ["boom"]

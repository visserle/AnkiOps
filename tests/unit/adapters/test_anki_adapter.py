"""Error-path tests for Anki.apply_note_changes."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ankiops.anki import Anki
from ankiops.anki_rpc import AnkiConnectionError
from ankiops.notes import Note
from ankiops.sync.report import Change, ChangeType


def _make_update_change() -> Change:
    return Change(
        ChangeType.UPDATE,
        101,
        "note_key: update",
        {"html_fields": {"Question": "Q", "Answer": "A2"}},
    )


def _make_delete_change() -> Change:
    return Change(ChangeType.DELETE, 202, "note_key: delete")


def _make_create_change(entity_repr: str = "note_key: create") -> Change:
    return Change(
        ChangeType.CREATE,
        None,
        entity_repr,
        {
            "note": Note(note_key=None, note_type="AnkiOpsQA", fields={}),
            "html_fields": {"Question": "Q", "Answer": "A"},
            "note_key": "created-key",
        },
    )


def test_apply_note_changes_collects_partial_errors():
    adapter = Anki()
    fake_non_create_result = [
        None,
        "change deck failed",
        "update failed",
        "delete failed",
    ]

    with patch(
        "ankiops.anki.invoke",
        side_effect=[
            fake_non_create_result,
            [{"canAdd": True}],
            ["create failed"],
        ],
    ):
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckX",
            needs_create_deck=True,
            creates=[_make_create_change()],
            updates=[_make_update_change()],
            deletes=[_make_delete_change()],
            cards_to_move=[1001],
        )

    assert created_ids == [None]
    assert len(errors) == 4
    assert any("Update failed (note_key: update)" in error for error in errors)
    assert any("Create failed (note_key: create)" in error for error in errors)
    assert any("Failed change_deck" in error for error in errors)
    assert any("Failed delete" in error for error in errors)


def test_apply_note_changes_surfaces_connection_exception():
    adapter = Anki()

    with patch("ankiops.anki.invoke", side_effect=AnkiConnectionError("boom")):
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


def test_apply_note_changes_preflight_reports_create_context():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        return_value=[
            {
                "canAdd": False,
                "error": "cannot create note because it is a duplicate",
            }
        ],
    ) as mock_invoke:
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckY",
            needs_create_deck=False,
            creates=[_make_create_change("'Duplicate Q...'")],
            updates=[],
            deletes=[],
            cards_to_move=[],
        )

    assert created_ids == [None]
    assert errors == [
        "Create failed ('Duplicate Q...'): cannot create note because it is a duplicate"
    ]
    assert [call.args[0] for call in mock_invoke.call_args_list] == [
        "canAddNotesWithErrorDetail"
    ]


def test_apply_note_changes_bulk_create_exception_keeps_create_context():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        side_effect=[
            [{"canAdd": True}],
            AnkiConnectionError("cannot create note because it is a duplicate"),
        ],
    ):
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckY",
            needs_create_deck=False,
            creates=[_make_create_change("'Duplicate Q...'")],
            updates=[],
            deletes=[],
            cards_to_move=[],
        )

    assert created_ids == [None]
    assert errors == [
        "Create failed ('Duplicate Q...'): cannot create note because it is a duplicate"
    ]


def test_fetch_notes_info_maps_tags():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        return_value=[
            {
                "noteId": 101,
                "modelName": "AnkiOpsQA",
                "fields": {
                    "Question": {"value": "Q"},
                    "Answer": {"value": "A"},
                },
                "cards": [1001],
                "tags": ["z", "a", "z"],
            }
        ],
    ):
        notes = adapter.fetch_notes_info([101])

    assert notes[101].tags == ("a", "z")


def test_change_notes_notetype_calls_ankiops_connect_action():
    adapter = Anki()

    with patch("ankiops.anki.invoke") as mock_invoke:
        adapter.change_notes_notetype([101, 102], "AnkiOpsQA", "shared/o/r/AnkiOpsQA")

    mock_invoke.assert_called_once_with(
        "changeNotesNotetype",
        noteIds=[101, 102],
        oldModel="AnkiOpsQA",
        newModel="shared/o/r/AnkiOpsQA",
    )


def test_fetch_note_ids_by_note_keys_searches_key_field_independent_of_model():
    adapter = Anki()

    with patch("ankiops.anki.invoke", return_value=[[101], [201, 202]]) as mock_invoke:
        note_ids = adapter.fetch_note_ids_by_note_keys({"key-b", "key-a"})

    assert note_ids == {"key-a": [101], "key-b": [201, 202]}
    assert mock_invoke.call_args.args == ("multi",)
    assert mock_invoke.call_args.kwargs["actions"] == [
        {"action": "findNotes", "params": {"query": '"AnkiOps Key:key-a"'}},
        {"action": "findNotes", "params": {"query": '"AnkiOps Key:key-b"'}},
    ]


def test_apply_note_changes_updates_fields_and_tags_with_update_note():
    adapter = Anki()
    update_change = Change(
        ChangeType.UPDATE,
        101,
        "note_key: update",
        {
            "html_fields": {"Question": "Q", "Answer": "A2"},
            "tags": ("z", "a"),
        },
    )

    with patch("ankiops.anki.invoke", return_value=[None]) as mock_invoke:
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckY",
            needs_create_deck=False,
            creates=[],
            updates=[update_change],
            deletes=[],
            cards_to_move=[],
        )

    assert created_ids == []
    assert errors == []
    action = mock_invoke.call_args.kwargs["actions"][0]
    assert action["action"] == "updateNote"
    assert action["params"]["note"] == {
        "id": 101,
        "fields": {"Question": "Q", "Answer": "A2"},
        "tags": ["a", "z"],
    }


def test_apply_note_changes_bulk_create_includes_tags():
    adapter = Anki()
    create_change = Change(
        ChangeType.CREATE,
        None,
        "note_key: create",
        {
            "note": Note(
                note_key=None,
                note_type="AnkiOpsQA",
                fields={},
                tags=("z", "a"),
            ),
            "html_fields": {"Question": "Q", "Answer": "A"},
            "note_key": "created-key",
        },
    )

    with patch(
        "ankiops.anki.invoke",
        side_effect=[
            [{"canAdd": True}],
            [303],
        ],
    ) as mock_invoke:
        created_ids, errors = adapter.apply_note_changes(
            deck_name="DeckY",
            needs_create_deck=False,
            creates=[create_change],
            updates=[],
            deletes=[],
            cards_to_move=[],
        )

    assert created_ids == [303]
    assert errors == []
    add_notes_call = mock_invoke.call_args_list[1]
    note_payload = add_notes_call.kwargs["notes"][0]
    assert note_payload["tags"] == ["a", "z"]


def test_fetch_note_type_states_normalizes_dict_styling_payload():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        return_value=[
            ["Term"],
            {"name": "MyType", "css": ".card { color: red; }"},
            {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
            [""],
            {},
        ],
    ):
        states = adapter.fetch_note_type_states(["MyType"])

    assert states["MyType"]["styling"] == ".card { color: red; }"


def test_fetch_note_type_states_preserves_string_styling_payload():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        return_value=[
            ["Term"],
            ".card { color: blue; }",
            {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
            [""],
            {},
        ],
    ):
        states = adapter.fetch_note_type_states(["MyType"])

    assert states["MyType"]["styling"] == ".card { color: blue; }"


def test_fetch_note_type_states_raises_on_malformed_styling_payload():
    adapter = Anki()

    with patch(
        "ankiops.anki.invoke",
        return_value=[
            ["Term"],
            {"name": "MyType"},
            {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
            [""],
            {},
        ],
    ):
        with pytest.raises(
            AnkiConnectionError,
            match="Malformed modelStyling response",
        ):
            adapter.fetch_note_type_states(["MyType"])

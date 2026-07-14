"""Error-path tests for Anki.apply_note_changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from ankiops.anki import Anki
from ankiops.anki_rpc import AnkiConnectionError
from ankiops.notes import Note
from ankiops.sync.report import Change, ChangeType


class _InvokeRecorder:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, action: str, **params):
        self.calls.append((action, params))
        if not self.results:
            return None
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


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
    fake_non_create_result = [
        None,
        "change deck failed",
        "update failed",
        "delete failed",
    ]
    invoke = _InvokeRecorder(
        fake_non_create_result,
        [{"canAdd": True}],
        ["create failed"],
    )
    adapter = Anki(invoke_func=invoke)

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
    adapter = Anki(invoke_func=_InvokeRecorder(AnkiConnectionError("boom")))

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
    invoke = _InvokeRecorder(
        [
            {
                "canAdd": False,
                "error": "cannot create note because it is a duplicate",
            }
        ]
    )
    adapter = Anki(invoke_func=invoke)

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
    assert [call[0] for call in invoke.calls] == ["canAddNotesWithErrorDetail"]


def test_apply_note_changes_bulk_create_exception_keeps_create_context():
    adapter = Anki(
        invoke_func=_InvokeRecorder(
            [{"canAdd": True}],
            AnkiConnectionError("cannot create note because it is a duplicate"),
        )
    )

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
    adapter = Anki(
        invoke_func=_InvokeRecorder(
            [
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
            ]
        )
    )

    notes = adapter.fetch_notes_info([101])

    assert notes[101].tags == ("a", "z")


def test_convert_notes_to_note_type_calls_ankiops_connect_action():
    invoke = _InvokeRecorder()
    adapter = Anki(invoke_func=invoke)

    adapter.convert_notes_to_note_type([101, 102], "AnkiOpsQA", "collab/o/r/AnkiOpsQA")

    assert invoke.calls == [
        (
            "convertNotesToNoteType",
            {
                "noteIds": [101, 102],
                "oldNoteType": "AnkiOpsQA",
                "newNoteType": "collab/o/r/AnkiOpsQA",
            },
        )
    ]


def test_fetch_note_ids_by_note_keys_searches_key_field_independent_of_model():
    invoke = _InvokeRecorder([[101], [201, 202]])
    adapter = Anki(invoke_func=invoke)

    note_ids = adapter.fetch_note_ids_by_note_keys({"key-b", "key-a"})

    assert note_ids == {"key-a": [101], "key-b": [201, 202]}
    action, params = invoke.calls[0]
    assert action == "multi"
    assert params["actions"] == [
        {"action": "findNotes", "params": {"query": '"AnkiOps Key:key-a"'}},
        {"action": "findNotes", "params": {"query": '"AnkiOps Key:key-b"'}},
    ]


def test_apply_note_changes_updates_fields_and_tags_with_update_note():
    invoke = _InvokeRecorder([None])
    adapter = Anki(invoke_func=invoke)
    update_change = Change(
        ChangeType.UPDATE,
        101,
        "note_key: update",
        {
            "html_fields": {"Question": "Q", "Answer": "A2"},
            "tags": ("z", "a"),
        },
    )

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
    action = invoke.calls[0][1]["actions"][0]
    assert action["action"] == "updateNote"
    assert action["params"]["note"] == {
        "id": 101,
        "fields": {"Question": "Q", "Answer": "A2"},
        "tags": ["a", "z"],
    }


def test_apply_note_changes_bulk_create_includes_tags():
    invoke = _InvokeRecorder(
        [{"canAdd": True}],
        [303],
    )
    adapter = Anki(invoke_func=invoke)
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
    note_payload = invoke.calls[1][1]["notes"][0]
    assert note_payload["tags"] == ["a", "z"]


def test_fetch_note_type_states_normalizes_dict_styling_payload():
    adapter = Anki(
        invoke_func=_InvokeRecorder(
            [
                ["Term"],
                {"name": "MyType", "css": ".card { color: red; }"},
                {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
                [""],
                {},
            ]
        )
    )

    states = adapter.fetch_note_type_states(["MyType"])

    assert states["MyType"]["styling"] == ".card { color: red; }"


def test_fetch_note_type_states_preserves_string_styling_payload():
    adapter = Anki(
        invoke_func=_InvokeRecorder(
            [
                ["Term"],
                ".card { color: blue; }",
                {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
                [""],
                {},
            ]
        )
    )

    states = adapter.fetch_note_type_states(["MyType"])

    assert states["MyType"]["styling"] == ".card { color: blue; }"


def test_fetch_note_type_states_raises_on_malformed_styling_payload():
    adapter = Anki(
        invoke_func=_InvokeRecorder(
            [
                ["Term"],
                {"name": "MyType"},
                {"Card 1": {"Front": "{{Term}}", "Back": "{{Term}}"}},
                [""],
                {},
            ]
        )
    )

    with pytest.raises(
        AnkiConnectionError,
        match="Malformed modelStyling response",
    ):
        adapter.fetch_note_type_states(["MyType"])


def test_adapter_media_operations_accept_regular_flat_media_files(tmp_path: Path):
    source_media = tmp_path / "source" / "media"
    source_media.mkdir(parents=True)
    source = source_media / "shared image.png"
    source.write_bytes(b"image")
    anki_media = tmp_path / "collection.media"
    anki_media.mkdir()
    adapter = Anki(invoke_func=_InvokeRecorder(str(anki_media)))

    adapter.push_media(source, source.name)
    assert (anki_media / source.name).read_bytes() == b"image"

    target_media = tmp_path / "target" / "media"
    target_media.mkdir(parents=True)
    target = target_media / source.name
    assert adapter.pull_media(source.name, target)
    assert target.read_bytes() == b"image"

    adapter.delete_media_file(source.name)
    assert not (anki_media / source.name).exists()

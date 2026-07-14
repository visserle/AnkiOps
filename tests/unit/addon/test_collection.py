from __future__ import annotations

import pytest

from anki_addon.actions import dispatch_action
from anki_addon.protocol import AnkiOpsConnectActionError
from tests.unit.addon.fakes import _ReadFakeCollection, _WriteFakeCollection


def test_dispatch_action_reads_collection_state():
    col = _ReadFakeCollection()

    assert dispatch_action(col, "getActiveProfile", {}) == "TestProfile"
    assert dispatch_action(col, "deckNamesAndIds", {}) == {
        "Default": 1,
        "Deck": 2,
    }
    assert dispatch_action(
        col,
        "findNotes",
        {"query": '"AnkiOps Key:key-101"'},
    ) == [101]
    assert dispatch_action(col, "notesInfo", {"notes": [101]}) == [
        {
            "noteId": 101,
            "modelName": "AnkiOpsQA",
            "fields": {
                "Question": {"value": "Q", "order": 0},
                "Answer": {"value": "A", "order": 1},
                "AnkiOps Key": {"value": "key-101", "order": 2},
            },
            "cards": [1001],
            "tags": ["z", "a"],
        }
    ]
    assert dispatch_action(col, "cardsInfo", {"cards": [1001]}) == [
        {
            "cardId": 1001,
            "note": 101,
            "deckName": "Deck",
            "modelName": "AnkiOpsQA",
        }
    ]
    assert dispatch_action(col, "findCards", {"query": "nid:101"}) == [1001]


def test_dispatch_action_imports_notes_and_media(tmp_path):
    col = _WriteFakeCollection(tmp_path / "collection.media")

    assert dispatch_action(col, "createDeck", {"deck": "Imported"}) == 3
    note_payload = {
        "deckName": "Imported",
        "modelName": "AnkiOpsQA",
        "fields": {
            "Question": "Q",
            "Answer": "A",
            "AnkiOps Key": "key-201",
        },
        "tags": ["z", "a"],
    }
    assert dispatch_action(
        col,
        "canAddNotesWithErrorDetail",
        {"notes": [note_payload]},
    ) == [{"canAdd": True}]

    created_ids = dispatch_action(
        col,
        "addNotes",
        {"notes": [note_payload]},
    )
    note_id = created_ids[0]

    assert (
        dispatch_action(
            col,
            "updateNote",
            {
                "note": {
                    "id": note_id,
                    "fields": {"Answer": "A2"},
                    "tags": ["updated"],
                }
            },
        )
        is None
    )
    assert (
        dispatch_action(col, "notesInfo", {"notes": [note_id]})[0]["fields"]["Answer"][
            "value"
        ]
        == "A2"
    )
    assert dispatch_action(col, "notesInfo", {"notes": [note_id]})[0]["tags"] == [
        "updated"
    ]

    note_info = dispatch_action(
        col,
        "notesInfo",
        {"notes": [note_id]},
    )[0]
    card_id = note_info["cards"][0]
    assert dispatch_action(col, "createDeck", {"deck": "Moved"}) == 4
    assert (
        dispatch_action(
            col,
            "changeDeck",
            {"cards": [card_id], "deck": "Moved"},
        )
        is None
    )
    assert (
        dispatch_action(col, "cardsInfo", {"cards": [card_id]})[0]["deckName"]
        == "Moved"
    )

    assert dispatch_action(col, "deleteNotes", {"notes": [note_id]}) is None
    assert dispatch_action(col, "notesInfo", {"notes": [note_id]}) == []
    assert dispatch_action(col, "createDeck", {"deck": "Obsolete"}) == 5
    assert (
        dispatch_action(
            col,
            "deleteDecks",
            {"decks": ["Obsolete"], "cardsToo": True},
        )
        is None
    )
    assert "Obsolete" not in dispatch_action(col, "deckNamesAndIds", {})
    assert dispatch_action(col, "getMediaDirPath", {}) == str(
        tmp_path / "collection.media"
    )


def test_delete_decks_requires_explicit_card_deletion_permission(tmp_path):
    col = _WriteFakeCollection(tmp_path / "collection.media")
    dispatch_action(col, "createDeck", {"deck": "Keep"})

    with pytest.raises(AnkiOpsConnectActionError, match="cardsToo=true"):
        dispatch_action(
            col,
            "deleteDecks",
            {"decks": ["Keep"], "cardsToo": False},
        )

    assert "Keep" in dispatch_action(col, "deckNamesAndIds", {})

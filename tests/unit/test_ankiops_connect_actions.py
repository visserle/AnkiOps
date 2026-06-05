from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_ANKIOPS_CONNECT_ACTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "anki_addon" / "ankiops_connect_actions.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ankiops_addon_ankiops_connect_actions",
    _ANKIOPS_CONNECT_ACTIONS_PATH,
)
ankiops_connect_actions = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(ankiops_connect_actions)

AnkiOpsConnectActionError = ankiops_connect_actions.AnkiOpsConnectActionError
change_notes_notetype = ankiops_connect_actions.change_notes_notetype
dispatch_ankiops_connect_action = (
    ankiops_connect_actions.dispatch_ankiops_connect_action
)


def _model(model_id: int, name: str, fields=None, templates=None):
    return {
        "id": model_id,
        "name": name,
        "css": ".card { color: black; }",
        "flds": [
            {"name": field, "description": "", "font": "Arial", "size": 20}
            for field in (fields or _FIELDS)
        ],
        "tmpls": [
            {"name": template, "qfmt": "{{Question}}", "afmt": "{{Answer}}"}
            for template in (templates or ["Card 1"])
        ],
    }


_FIELDS = ["Question", "Answer", "AnkiOps Key"]


class _FakeNote:
    def __init__(
        self,
        note_id: int,
        model: dict,
        fields: dict[str, str],
        tags=(),
    ):
        self.id = note_id
        self._model = model
        self.tags = list(tags)
        self.fields = [
            fields.get(field["name"], "")
            for field in model["flds"]
        ]

    def note_type(self):
        return self._model


class _FakeModels:
    def __init__(self, collection):
        self.collection = collection

    def by_name(self, name: str):
        return self.collection.models_by_name.get(name)

    def all_names(self):
        return list(self.collection.models_by_name)

    def new(self, name: str):
        model_id = max(self.collection.models_by_id, default=0) + 1
        return {
            "id": model_id,
            "name": name,
            "flds": [],
            "tmpls": [],
            "css": "",
        }

    def new_field(self, name: str):
        return {"name": name, "description": "", "font": "Arial", "size": 20}

    def add_field(self, model, field):
        model["flds"].append(field)

    def new_template(self, name: str):
        return {"name": name, "qfmt": "", "afmt": ""}

    def add_template(self, model, template):
        model["tmpls"].append(template)

    def add(self, model):
        self.collection.models_by_name[model["name"]] = model
        self.collection.models_by_id[model["id"]] = model

    def save(self, model):
        self.collection.models_by_name[model["name"]] = model
        self.collection.models_by_id[model["id"]] = model

    def change_notetype_info(self, old_notetype_id: int, new_notetype_id: int):
        self.collection.pending_new_model = self.collection.models_by_id[
            new_notetype_id
        ]
        return SimpleNamespace(
            input=SimpleNamespace(note_ids=[], new_fields=[], new_templates=[]),
        )

    def change_notetype_of_notes(self, request):
        new_model = self.collection.pending_new_model
        for note_id in request.note_ids:
            note = self.collection.notes[note_id]
            old_values = list(note.fields)
            note.fields = [
                old_values[old_index]
                for old_index in request.new_fields
            ]
            note._model = new_model
            if self.collection.mutate_card_due_on_change:
                for card in self.collection.cards.values():
                    if card["nid"] == note_id:
                        card["due"] += 1


class _FakeCollection:
    def __init__(self):
        self.old_model = _model(1, "AnkiOpsQA")
        self.new_model = _model(2, "collab/owner/repo/AnkiOpsQA")
        self.models_by_name = {
            self.old_model["name"]: self.old_model,
            self.new_model["name"]: self.new_model,
        }
        self.models_by_id = {
            self.old_model["id"]: self.old_model,
            self.new_model["id"]: self.new_model,
        }
        self.models = _FakeModels(self)
        self.notes = {}
        self.cards = {}
        self.pending_new_model = None
        self.mutate_card_due_on_change = False

    def add_note(self, note_id: int, model: dict, key: str):
        self.notes[note_id] = _FakeNote(
            note_id,
            model,
            {
                "Question": f"Q{note_id}",
                "Answer": f"A{note_id}",
                "AnkiOps Key": key,
            },
        )
        self.cards[note_id * 10] = {
            "id": note_id * 10,
            "nid": note_id,
            "ord": 0,
            "did": 1,
            "queue": 2,
            "type": 2,
            "due": 10,
            "ivl": 5,
            "factor": 2500,
            "reps": 3,
            "lapses": 0,
            "left": 0,
            "odue": 0,
            "odid": 0,
        }

    def get_note(self, note_id: int):
        if note_id not in self.notes:
            raise KeyError(note_id)
        return self.notes[note_id]

    def ankiops_connect_cards_snapshot(self, note_ids):
        return {
            card_id: tuple(card.values())
            for card_id, card in self.cards.items()
            if card["nid"] in note_ids
        }


class _FakeDecks:
    def __init__(self):
        self.names_by_id = {1: "Default", 2: "Deck"}
        self.next_id = 3

    def all_names_and_ids(self):
        return [
            SimpleNamespace(name=name, id=deck_id)
            for deck_id, name in self.names_by_id.items()
        ]

    def id(self, name):
        for deck_id, deck_name in self.names_by_id.items():
            if deck_name == name:
                return deck_id
        deck_id = self.next_id
        self.next_id += 1
        self.names_by_id[deck_id] = name
        return deck_id

    def name(self, deck_id):
        return self.names_by_id[deck_id]


class _FakeCard:
    def __init__(self, card_id: int, note_id: int, deck_id: int):
        self.id = card_id
        self.nid = note_id
        self.did = deck_id


class _ReadFakeCollection:
    ankiops_connect_active_profile = "TestProfile"

    def __init__(self):
        self.model = _model(1, "AnkiOpsQA")
        self.decks = _FakeDecks()
        self.notes = {
            101: _FakeNote(
                101,
                self.model,
                {"Question": "Q", "Answer": "A", "AnkiOps Key": "key-101"},
                tags=["z", "a"],
            )
        }
        self.cards = {1001: _FakeCard(1001, 101, 2)}

    def find_notes(self, query: str):
        if query == 'note:"AnkiOpsQA"' or query == '"AnkiOps Key:key-101"':
            return [101]
        return []

    def find_cards(self, query: str):
        if query == "nid:101":
            return [1001]
        return []

    def get_note(self, note_id: int):
        return self.notes[note_id]

    def get_card(self, card_id: int):
        return self.cards[card_id]


class _WriteFakeCollection(_FakeCollection):
    def __init__(self, media_dir: Path):
        super().__init__()
        self.decks = _FakeDecks()
        self.media = SimpleNamespace(dir=lambda: str(media_dir))
        self.next_note_id = 201
        self.next_card_id = 2001

    def new_note(self, model):
        return _FakeNote(0, model, {})

    def add_note(self, note, deck_id: int):
        note.id = self.next_note_id
        self.next_note_id += 1
        self.notes[note.id] = note
        card_id = self.next_card_id
        self.next_card_id += 1
        self.cards[card_id] = _FakeCard(card_id, note.id, deck_id)
        return note.id

    def update_note(self, note):
        self.notes[note.id] = note

    def remove_notes(self, note_ids):
        for note_id in note_ids:
            self.notes.pop(note_id, None)
        self.cards = {
            card_id: card
            for card_id, card in self.cards.items()
            if card.nid not in note_ids
        }

    def find_cards(self, query: str):
        if not query.startswith("nid:"):
            return []
        note_id = int(query.split(":", 1)[1])
        return [
            card_id
            for card_id, card in self.cards.items()
            if card.nid == note_id
        ]

    def get_card(self, card_id: int):
        return self.cards[card_id]


def test_change_notes_notetype_converts_selected_notes_only():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")
    col.add_note(102, col.old_model, "key-102")

    result = change_notes_notetype(
        col,
        [101],
        "AnkiOpsQA",
        "collab/owner/repo/AnkiOpsQA",
    )

    assert result == {"changed": 1}
    assert col.notes[101].note_type()["name"] == "collab/owner/repo/AnkiOpsQA"
    assert col.notes[102].note_type()["name"] == "AnkiOpsQA"


def test_change_notes_notetype_blocks_missing_model():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="New note type not found"):
        change_notes_notetype(col, [101], "AnkiOpsQA", "missing")


def test_change_notes_notetype_blocks_wrong_current_model():
    col = _FakeCollection()
    col.add_note(101, col.new_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="expected 'AnkiOpsQA'"):
        change_notes_notetype(
            col,
            [101],
            "AnkiOpsQA",
            "collab/owner/repo/AnkiOpsQA",
        )


def test_change_notes_notetype_blocks_missing_ankiops_key_field():
    col = _FakeCollection()
    col.old_model = _model(1, "AnkiOpsQA", fields=["Question", "Answer"])
    col.models_by_name["AnkiOpsQA"] = col.old_model
    col.models_by_id[1] = col.old_model
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="lacks AnkiOps Key"):
        change_notes_notetype(
            col,
            [101],
            "AnkiOpsQA",
            "collab/owner/repo/AnkiOpsQA",
        )


def test_change_notes_notetype_blocks_mismatched_fields():
    col = _FakeCollection()
    col.new_model = _model(
        2,
        "collab/owner/repo/AnkiOpsQA",
        fields=["Question", "Different", "AnkiOps Key"],
    )
    col.models_by_name[col.new_model["name"]] = col.new_model
    col.models_by_id[2] = col.new_model
    col.add_note(101, col.old_model, "key-101")

    with pytest.raises(AnkiOpsConnectActionError, match="field names differ"):
        change_notes_notetype(
            col,
            [101],
            "AnkiOpsQA",
            "collab/owner/repo/AnkiOpsQA",
        )


def test_change_notes_notetype_blocks_post_verification_failures():
    col = _FakeCollection()
    col.add_note(101, col.old_model, "key-101")
    col.mutate_card_due_on_change = True

    with pytest.raises(
        AnkiOpsConnectActionError,
        match="Post-conversion verification failed",
    ):
        change_notes_notetype(
            col,
            [101],
            "AnkiOpsQA",
            "collab/owner/repo/AnkiOpsQA",
        )


def test_dispatch_ankiops_connect_action_routes_version():
    assert dispatch_ankiops_connect_action(None, "version", {}) == 1


def test_dispatch_ankiops_connect_action_reads_collection_state():
    col = _ReadFakeCollection()

    assert dispatch_ankiops_connect_action(col, "getActiveProfile", {}) == "TestProfile"
    assert dispatch_ankiops_connect_action(col, "deckNamesAndIds", {}) == {
        "Default": 1,
        "Deck": 2,
    }
    assert dispatch_ankiops_connect_action(
        col,
        "findNotes",
        {"query": '"AnkiOps Key:key-101"'},
    ) == [101]
    assert dispatch_ankiops_connect_action(col, "notesInfo", {"notes": [101]}) == [
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
    assert dispatch_ankiops_connect_action(col, "cardsInfo", {"cards": [1001]}) == [
        {
            "cardId": 1001,
            "note": 101,
            "deckName": "Deck",
            "modelName": "AnkiOpsQA",
        }
    ]


def test_dispatch_ankiops_connect_action_reads_model_state():
    col = _FakeCollection()
    col.old_model["flds"][0]["description"] = "Prompt"
    col.old_model["flds"][0]["size"] = 14

    assert dispatch_ankiops_connect_action(col, "modelNames", {}) == [
        "AnkiOpsQA",
        "collab/owner/repo/AnkiOpsQA",
    ]
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsQA"},
    ) == ["Question", "Answer", "AnkiOps Key"]
    assert dispatch_ankiops_connect_action(
        col,
        "modelStyling",
        {"modelName": "AnkiOpsQA"},
    ) == {"name": "AnkiOpsQA", "css": ".card { color: black; }"}
    assert dispatch_ankiops_connect_action(
        col,
        "modelTemplates",
        {"modelName": "AnkiOpsQA"},
    ) == {"Card 1": {"Front": "{{Question}}", "Back": "{{Answer}}"}}
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldDescriptions",
        {"modelName": "AnkiOpsQA"},
    ) == ["Prompt", "", ""]
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldFonts",
        {"modelName": "AnkiOpsQA"},
    ) == {
        "Question": {"font": "Arial", "size": 14},
        "Answer": {"font": "Arial", "size": 20},
        "AnkiOps Key": {"font": "Arial", "size": 20},
    }


def test_dispatch_ankiops_connect_action_creates_and_updates_model_state():
    col = _FakeCollection()

    assert dispatch_ankiops_connect_action(
        col,
        "createModel",
        {
            "modelName": "AnkiOpsNew",
            "inOrderFields": ["Question", "Answer"],
            "css": ".card { color: green; }",
            "isCloze": False,
            "cardTemplates": [
                {"Name": "Card 1", "Front": "{{Question}}", "Back": "{{Answer}}"}
            ],
        },
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsNew"},
    ) == ["Question", "Answer"]

    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldAdd",
        {"modelName": "AnkiOpsNew", "fieldName": "Extra"},
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldReposition",
        {"modelName": "AnkiOpsNew", "fieldName": "Extra", "index": 1},
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldSetDescription",
        {"modelName": "AnkiOpsNew", "fieldName": "Extra", "description": "Details"},
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldSetFontSize",
        {"modelName": "AnkiOpsNew", "fieldName": "Extra", "fontSize": 15},
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "updateModelStyling",
        {"model": {"name": "AnkiOpsNew", "css": ".card { color: blue; }"}},
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelTemplateRename",
        {
            "modelName": "AnkiOpsNew",
            "oldTemplateName": "Card 1",
            "newTemplateName": "Review",
        },
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "modelTemplateAdd",
        {
            "modelName": "AnkiOpsNew",
            "template": {"Name": "Extra", "Front": "{{Extra}}", "Back": "{{Answer}}"},
        },
    ) is None
    assert dispatch_ankiops_connect_action(
        col,
        "updateModelTemplates",
        {
            "model": {
                "name": "AnkiOpsNew",
                "templates": {
                    "Review": {"Front": "{{Question}}?", "Back": "{{Answer}}!"},
                    "Extra": {"Front": "{{Extra}}?", "Back": "{{Answer}}!"},
                },
            }
        },
    ) is None

    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldNames",
        {"modelName": "AnkiOpsNew"},
    ) == ["Question", "Extra", "Answer"]
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldDescriptions",
        {"modelName": "AnkiOpsNew"},
    ) == ["", "Details", ""]
    assert dispatch_ankiops_connect_action(
        col,
        "modelFieldFonts",
        {"modelName": "AnkiOpsNew"},
    )["Extra"]["size"] == 15
    assert dispatch_ankiops_connect_action(
        col,
        "modelStyling",
        {"modelName": "AnkiOpsNew"},
    )["css"] == ".card { color: blue; }"
    assert dispatch_ankiops_connect_action(
        col,
        "modelTemplates",
        {"modelName": "AnkiOpsNew"},
    ) == {
        "Review": {"Front": "{{Question}}?", "Back": "{{Answer}}!"},
        "Extra": {"Front": "{{Extra}}?", "Back": "{{Answer}}!"},
    }


def test_dispatch_ankiops_connect_action_imports_notes_and_media(tmp_path):
    col = _WriteFakeCollection(tmp_path / "collection.media")

    assert dispatch_ankiops_connect_action(col, "createDeck", {"deck": "Imported"}) == 3
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
    assert dispatch_ankiops_connect_action(
        col,
        "canAddNotesWithErrorDetail",
        {"notes": [note_payload]},
    ) == [{"canAdd": True}]

    created_ids = dispatch_ankiops_connect_action(
        col,
        "addNotes",
        {"notes": [note_payload]},
    )
    note_id = created_ids[0]

    assert dispatch_ankiops_connect_action(
        col,
        "updateNote",
        {
            "note": {
                "id": note_id,
                "fields": {"Answer": "A2"},
                "tags": ["updated"],
            }
        },
    ) is None
    assert dispatch_ankiops_connect_action(col, "notesInfo", {"notes": [note_id]})[0][
        "fields"
    ]["Answer"]["value"] == "A2"
    assert dispatch_ankiops_connect_action(col, "notesInfo", {"notes": [note_id]})[0][
        "tags"
    ] == ["updated"]

    note_info = dispatch_ankiops_connect_action(
        col,
        "notesInfo",
        {"notes": [note_id]},
    )[0]
    card_id = note_info["cards"][0]
    assert dispatch_ankiops_connect_action(col, "createDeck", {"deck": "Moved"}) == 4
    assert dispatch_ankiops_connect_action(
        col,
        "changeDeck",
        {"cards": [card_id], "deck": "Moved"},
    ) is None
    assert dispatch_ankiops_connect_action(col, "cardsInfo", {"cards": [card_id]})[0][
        "deckName"
    ] == "Moved"

    assert (
        dispatch_ankiops_connect_action(col, "deleteNotes", {"notes": [note_id]})
        is None
    )
    assert dispatch_ankiops_connect_action(col, "notesInfo", {"notes": [note_id]}) == []
    assert dispatch_ankiops_connect_action(col, "getMediaDirPath", {}) == str(
        tmp_path / "collection.media"
    )


def test_dispatch_ankiops_connect_action_multi_collects_results_and_errors():
    results = dispatch_ankiops_connect_action(
        None,
        "multi",
        {
            "actions": [
                {"action": "version", "params": {}},
                {"action": "missing", "params": {}},
            ]
        },
    )

    assert results == [1, "Unknown AnkiOpsConnect action: missing"]


def test_dispatch_ankiops_connect_action_blocks_unknown_action():
    with pytest.raises(
        AnkiOpsConnectActionError,
        match="Unknown AnkiOpsConnect action",
    ):
        dispatch_ankiops_connect_action(None, "missing", {})

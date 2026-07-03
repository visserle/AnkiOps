from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


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
        self.fields = [fields.get(field["name"], "") for field in model["flds"]]

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
            note.fields = [old_values[old_index] for old_index in request.new_fields]
            note._model = new_model
            if self.collection.mutate_card_due_on_change:
                for card in self.collection.cards.values():
                    if card.nid == note_id:
                        card.due += 1


class _FakeCollection:
    def __init__(self):
        self.old_model = _model(1, "AnkiOpsQA")
        self.new_model = _model(2, "shared/owner/repo/AnkiOpsQA")
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
        self.cards[note_id * 10] = _FakeCard(note_id * 10, note_id, 1)

    def get_note(self, note_id: int):
        if note_id not in self.notes:
            raise KeyError(note_id)
        return self.notes[note_id]

    def ankiops_connect_cards_snapshot(self, note_ids):
        return {
            card_id: card.snapshot()
            for card_id, card in self.cards.items()
            if card.nid in note_ids
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
        self.ord = 0
        self.did = deck_id
        self.queue = 2
        self.type = 2
        self.due = 10
        self.ivl = 5
        self.factor = 2500
        self.reps = 3
        self.lapses = 0
        self.left = 0
        self.odue = 0
        self.odid = 0

    def snapshot(self):
        return (
            self.id,
            self.nid,
            self.ord,
            self.did,
            self.queue,
            self.type,
            self.due,
            self.ivl,
            self.factor,
            self.reps,
            self.lapses,
            self.left,
            self.odue,
            self.odid,
        )


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
        return [card_id for card_id, card in self.cards.items() if card.nid == note_id]

    def get_card(self, card_id: int):
        return self.cards[card_id]

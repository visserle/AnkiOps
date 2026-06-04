from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_BRIDGE_ACTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "anki_addon" / "bridge_actions.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ankiops_addon_bridge_actions",
    _BRIDGE_ACTIONS_PATH,
)
bridge_actions = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(bridge_actions)

BridgeActionError = bridge_actions.BridgeActionError
change_notes_notetype = bridge_actions.change_notes_notetype
dispatch_bridge_action = bridge_actions.dispatch_bridge_action


def _model(model_id: int, name: str, fields=None, templates=None):
    return {
        "id": model_id,
        "name": name,
        "flds": [{"name": field} for field in (fields or _FIELDS)],
        "tmpls": [{"name": template} for template in (templates or ["Card 1"])],
    }


_FIELDS = ["Question", "Answer", "AnkiOps Key"]


class _FakeNote:
    def __init__(self, note_id: int, model: dict, fields: dict[str, str]):
        self.id = note_id
        self._model = model
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

    def ankiops_bridge_cards_snapshot(self, note_ids):
        return {
            card_id: tuple(card.values())
            for card_id, card in self.cards.items()
            if card["nid"] in note_ids
        }


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

    with pytest.raises(BridgeActionError, match="New note type not found"):
        change_notes_notetype(col, [101], "AnkiOpsQA", "missing")


def test_change_notes_notetype_blocks_wrong_current_model():
    col = _FakeCollection()
    col.add_note(101, col.new_model, "key-101")

    with pytest.raises(BridgeActionError, match="expected 'AnkiOpsQA'"):
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

    with pytest.raises(BridgeActionError, match="lacks AnkiOps Key"):
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

    with pytest.raises(BridgeActionError, match="field names differ"):
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

    with pytest.raises(BridgeActionError, match="Post-conversion verification failed"):
        change_notes_notetype(
            col,
            [101],
            "AnkiOpsQA",
            "collab/owner/repo/AnkiOpsQA",
        )


def test_dispatch_bridge_action_routes_version():
    assert dispatch_bridge_action(None, "version", {}) == {"version": 1}


def test_dispatch_bridge_action_blocks_unknown_action():
    with pytest.raises(BridgeActionError, match="Unknown AnkiOps bridge action"):
        dispatch_bridge_action(None, "missing", {})

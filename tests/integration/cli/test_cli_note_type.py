"""CLI behavior tests for note-types command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from ankiops.cli import main
from ankiops.note_types_command import run as run_note_type
from tests.support.deck_files import DeckFileHarness


def _seed_note_types(note_types_dir):
    DeckFileHarness().eject_default_note_types(note_types_dir)


class _FakeNoteTypeAnki:
    def __init__(self, model_names=None, model_states=None):
        self.model_names = model_names or []
        self.model_states = model_states or {}

    def get_active_profile(self):
        return "TestProfile"

    def fetch_note_type_names(self):
        return self.model_names

    def fetch_note_type_states(self, model_names):
        return {name: self.model_states[name] for name in model_names}


def test_note_types_import_writes_files_and_summary(tmp_path, caplog):
    _seed_note_types(tmp_path / "note_types")

    fake_anki = _FakeNoteTypeAnki(
        ["MyType"],
        {
            "MyType": {
                "fields": ["Term", "Definition", "Choice 1"],
                "styling": {"name": "MyType", "css": ".card { color: red; }"},
                "templates": {
                    "Card 1": {"Front": "{{Term}}", "Back": "{{Definition}}"},
                    "Card 2": {"Front": "{{Definition}}", "Back": "{{Term}}"},
                },
                "descriptions": {},
                "fonts": {},
            }
        },
    )

    args = SimpleNamespace(action="list", add_name="MyType")

    with (
        patch("ankiops.note_types_command.connect_or_exit", return_value=fake_anki),
        patch(
            "ankiops.note_types_command.require_collection_root",
            return_value=tmp_path,
        ),
        patch(
            "builtins.input",
            side_effect=["TM", "y", "D", "y", "DEF", "y", "X1", "y"],
        ),
        caplog.at_level("INFO"),
    ):
        run_note_type(args)

    note_type_dir = tmp_path / "note_types" / "MyType"
    assert (note_type_dir / "note_type.yaml").exists()
    assert (note_type_dir / "Styling.css").exists()
    assert (note_type_dir / "Front.template.anki").exists()
    assert (note_type_dir / "Back.template.anki").exists()
    assert (note_type_dir / "Front2.template.anki").exists()
    assert (note_type_dir / "Back2.template.anki").exists()
    assert (note_type_dir / "Styling.css").read_text("utf-8") == ".card { color: red; }"

    payload = yaml.safe_load((note_type_dir / "note_type.yaml").read_text("utf-8"))
    assert "name" not in payload
    assert payload["is_choice"] is True
    assert payload["fields"] == [
        {"name": "Term", "label": "TM:", "identifying": True},
        {"name": "Definition", "label": "DEF:", "identifying": True},
        {"name": "Choice 1", "label": "X1:", "identifying": True},
    ]


def test_note_types_import_reprompts_on_identifying_label_conflict(tmp_path, caplog):
    _seed_note_types(tmp_path / "note_types")

    fake_anki = _FakeNoteTypeAnki(
        ["MyType"],
        {
            "MyType": {
                "fields": ["Question", "Context"],
                "styling": ".card { color: red; }",
                "templates": {
                    "Card 1": {"Front": "{{Question}}", "Back": "{{Context}}"},
                },
                "descriptions": {},
                "fonts": {},
            }
        },
    )
    args = SimpleNamespace(action="list", add_name="MyType")

    with (
        patch("ankiops.note_types_command.connect_or_exit", return_value=fake_anki),
        patch(
            "ankiops.note_types_command.require_collection_root",
            return_value=tmp_path,
        ),
        patch("builtins.input", side_effect=["Q", "n", "Q", "y", "CTX", "y"]),
        caplog.at_level("INFO"),
    ):
        run_note_type(args)

    payload = yaml.safe_load(
        (tmp_path / "note_types" / "MyType" / "note_type.yaml").read_text("utf-8")
    )
    assert payload["fields"][0] == {
        "name": "Question",
        "label": "Q:",
        "identifying": True,
    }


def test_note_types_import_reprompts_on_invalid_label(tmp_path, caplog):
    _seed_note_types(tmp_path / "note_types")

    fake_anki = _FakeNoteTypeAnki(
        ["MyType"],
        {
            "MyType": {
                "fields": ["Question", "Answer"],
                "styling": ".card { color: red; }",
                "templates": {
                    "Card 1": {"Front": "{{Question}}", "Back": "{{Answer}}"},
                },
                "descriptions": {},
                "fonts": {},
            }
        },
    )
    args = SimpleNamespace(action="list", add_name="MyType")

    with (
        patch("ankiops.note_types_command.connect_or_exit", return_value=fake_anki),
        patch(
            "ankiops.note_types_command.require_collection_root",
            return_value=tmp_path,
        ),
        patch("builtins.input", side_effect=["A B", "QQ", "y", "AA", "y"]),
        caplog.at_level("ERROR"),
    ):
        run_note_type(args)

    payload = yaml.safe_load(
        (tmp_path / "note_types" / "MyType" / "note_type.yaml").read_text("utf-8")
    )
    assert "Label must start with a letter" in caplog.text
    assert payload["fields"][:2] == [
        {"name": "Question", "label": "QQ:", "identifying": True},
        {"name": "Answer", "label": "AA:", "identifying": True},
    ]


def test_note_types_import_rejects_unknown_model(tmp_path):
    fake_anki = _FakeNoteTypeAnki(["AnkiOpsQA"])

    with (
        patch("ankiops.note_types_command.connect_or_exit", return_value=fake_anki),
        patch(
            "ankiops.note_types_command.require_collection_root",
            return_value=tmp_path,
        ),
        patch("sys.argv", ["ankiops", "note-types", "--add", "DoesNotExist"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


def test_note_types_import_rejects_existing_target_folder(tmp_path):
    existing = tmp_path / "note_types" / "MyType"
    existing.mkdir(parents=True, exist_ok=True)

    fake_anki = _FakeNoteTypeAnki(["MyType"])

    with (
        patch("ankiops.note_types_command.connect_or_exit", return_value=fake_anki),
        patch(
            "ankiops.note_types_command.require_collection_root",
            return_value=tmp_path,
        ),
        patch("sys.argv", ["ankiops", "note-types", "--add", "MyType"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1

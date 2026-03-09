"""CLI behavior tests for note-type command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ankiops.cli import main
from ankiops.fs import FileSystemAdapter
from ankiops.note_type_cli import run as run_note_type


def _seed_note_types(note_types_dir):
    FileSystemAdapter().eject_builtin_note_types(note_types_dir)


def test_note_type_info_logs_label_inventory(tmp_path, caplog):
    (tmp_path / ".ankiops.db").write_text("", encoding="utf-8")
    _seed_note_types(tmp_path / "note_types")
    args = SimpleNamespace(info=True, name=None)

    with (
        patch(
            "ankiops.note_type_cli.require_initialized_collection_dir",
            return_value=tmp_path,
        ),
        patch(
            "ankiops.note_type_cli.get_note_types_dir",
            return_value=tmp_path / "note_types",
        ),
        caplog.at_level("INFO"),
    ):
        run_note_type(args)

    assert "Taken labels:" in caplog.text
    assert "C2:  [IDENTIFYING] -> AnkiOpsChoice.Choice 2" in caplog.text
    assert "AI:   [IDENTIFYING] ->" not in caplog.text
    assert "Free labels: any valid label not listed above." in caplog.text
    assert "By note type:" in caplog.text
    assert "IDENTIFYING rule: base IDENTIFYING labels" in caplog.text
    assert "IDENTIFYING labels: T:" in caplog.text


def test_note_type_rejects_info_with_name():
    with patch("sys.argv", ["ankiops", "note-type", "AnkiOpsQA", "--info"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2


def test_note_type_rejects_missing_name_without_info():
    with patch("sys.argv", ["ankiops", "note-type"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2


def test_note_type_copy_writes_files_and_summary(tmp_path, caplog):
    _seed_note_types(tmp_path / "note_types")

    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    fake_anki.fetch_model_names.return_value = ["MyType"]
    fake_anki.fetch_model_states.return_value = {
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
    }

    args = SimpleNamespace(info=False, name="MyType")

    with (
        patch("ankiops.note_type_cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.note_type_cli.require_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.note_type_cli.get_note_types_dir",
            return_value=tmp_path / "note_types",
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

    assert "Copied note type 'MyType'" in caplog.text
    assert "Saved to:" in caplog.text


def test_note_type_copy_reprompts_on_identifying_label_conflict(tmp_path, caplog):
    _seed_note_types(tmp_path / "note_types")

    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    fake_anki.fetch_model_names.return_value = ["MyType"]
    fake_anki.fetch_model_states.return_value = {
        "MyType": {
            "fields": ["Question", "Context"],
            "styling": ".card { color: red; }",
            "templates": {
                "Card 1": {"Front": "{{Question}}", "Back": "{{Context}}"},
            },
            "descriptions": {},
            "fonts": {},
        }
    }
    args = SimpleNamespace(info=False, name="MyType")

    with (
        patch("ankiops.note_type_cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.note_type_cli.require_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.note_type_cli.get_note_types_dir",
            return_value=tmp_path / "note_types",
        ),
        patch("builtins.input", side_effect=["Q", "n", "Q", "y", "CTX", "y"]),
        caplog.at_level("INFO"),
    ):
        run_note_type(args)

    assert "Label 'Q:' already has IDENTIFYING=True" in caplog.text
    payload = yaml.safe_load(
        (tmp_path / "note_types" / "MyType" / "note_type.yaml").read_text("utf-8")
    )
    assert payload["fields"][0] == {
        "name": "Question",
        "label": "Q:",
        "identifying": True,
    }


def test_note_type_copy_rejects_unknown_model(tmp_path):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    fake_anki.fetch_model_names.return_value = ["AnkiOpsQA"]

    with (
        patch("ankiops.note_type_cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.note_type_cli.require_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.note_type_cli.get_note_types_dir",
            return_value=tmp_path / "note_types",
        ),
        patch("sys.argv", ["ankiops", "note-type", "DoesNotExist"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


def test_note_type_copy_rejects_existing_target_folder(tmp_path):
    existing = tmp_path / "note_types" / "MyType"
    existing.mkdir(parents=True, exist_ok=True)

    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"

    with (
        patch("ankiops.note_type_cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.note_type_cli.require_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.note_type_cli.get_note_types_dir",
            return_value=tmp_path / "note_types",
        ),
        patch("sys.argv", ["ankiops", "note-type", "MyType"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1

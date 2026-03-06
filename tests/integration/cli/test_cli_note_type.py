"""CLI behavior tests for note-type command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ankiops.cli import main, run_note_type


def test_note_type_info_logs_prefix_inventory(tmp_path, caplog):
    (tmp_path / ".ankiops.db").write_text("", encoding="utf-8")
    args = SimpleNamespace(info=True, name=None)

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.get_note_types_dir", return_value=tmp_path / "note_types"),
        caplog.at_level("INFO"),
    ):
        run_note_type(args)

    assert "Taken prefixes:" in caplog.text
    assert "By note type:" in caplog.text
    assert "IDENTIFYING rule: base IDENTIFYING prefixes" in caplog.text
    assert "C2:  -> AnkiOpsChoice.Choice 2 [IDENTIFYING]" in caplog.text
    assert (
        "AnkiOpsChoice: Q:, C1:, C2:, C3:, C4:, C5:, C6:, C7:, C8:, A:, E:, M:, S:, "
        "AI:"
    ) in caplog.text
    assert (
        "AnkiOpsChoice: Q: [IDENTIFYING], C1: [IDENTIFYING], C2: [IDENTIFYING]"
        not in caplog.text
    )
    assert "AnkiOpsCloze: T:, E:, M:, S:, AI:" in caplog.text
    assert "AnkiOpsCloze: T: [IDENTIFYING]" not in caplog.text
    assert "required" not in caplog.text.lower()


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
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    fake_anki.fetch_model_names.return_value = ["MyType"]
    fake_anki.fetch_model_states.return_value = {
        "MyType": {
            "fields": ["Term", "Definition", "Choice 1"],
            "styling": ".card { color: red; }",
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
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.get_note_types_dir", return_value=tmp_path / "note_types"),
        patch(
            "builtins.input",
            side_effect=["TM:", "y", "D:", "y", "X1:", "n"],
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

    payload = yaml.safe_load((note_type_dir / "note_type.yaml").read_text("utf-8"))
    assert payload["name"] == "MyType"
    assert payload["is_choice"] is True
    assert payload["fields"] == [
        {"name": "Term", "prefix": "TM:", "identifying": True},
        {"name": "Definition", "prefix": "D:", "identifying": True},
        {"name": "Choice 1", "prefix": "X1:", "identifying": False},
    ]

    assert "Copied note type 'MyType'" in caplog.text
    assert "Saved to:" in caplog.text


def test_note_type_copy_rejects_unknown_model(tmp_path):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    fake_anki.fetch_model_names.return_value = ["AnkiOpsQA"]

    with (
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.get_note_types_dir", return_value=tmp_path / "note_types"),
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
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.get_note_types_dir", return_value=tmp_path / "note_types"),
        patch("sys.argv", ["ankiops", "note-type", "MyType"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1

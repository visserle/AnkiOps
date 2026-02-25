"""Tests for note type sync behavior and cache usage."""

from __future__ import annotations

from unittest.mock import MagicMock

from ankiops.db import SQLiteDbAdapter
from ankiops.models import Field, NoteTypeConfig
from ankiops.sync_note_types import sync_note_types


def _qa_config(*, css: str) -> NoteTypeConfig:
    return NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[
            Field(name="Question", prefix="Q:", identifying=True),
            Field(name="Answer", prefix="A:", identifying=True),
        ],
        css=css,
        templates=[{"Name": "Card 1", "Front": "{{Question}}", "Back": "{{Answer}}"}],
    )


def test_sync_note_types_uses_cache_when_unchanged(tmp_path):
    fs = MagicMock()
    fs.load_note_type_configs.return_value = [_qa_config(css=".card { color: black; }")]

    anki = MagicMock()
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    anki.fetch_model_states.return_value = {"AnkiOpsQA": {}}

    db = SQLiteDbAdapter.load(tmp_path)
    try:
        first = sync_note_types(anki, fs, tmp_path, db)
        assert first == "1 synced"
        assert anki.fetch_model_states.call_count == 1
        assert anki.update_models.call_count == 1

        anki.fetch_model_states.reset_mock()
        anki.update_models.reset_mock()

        second = sync_note_types(anki, fs, tmp_path, db)
        assert second == "1 up to date (cached)"
        anki.fetch_model_states.assert_not_called()
        anki.update_models.assert_not_called()
    finally:
        db.close()


def test_sync_note_types_cache_invalidates_on_local_change(tmp_path):
    fs = MagicMock()
    fs.load_note_type_configs.side_effect = [
        [_qa_config(css=".card { color: black; }")],
        [_qa_config(css=".card { color: blue; }")],
    ]

    anki = MagicMock()
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    anki.fetch_model_states.return_value = {"AnkiOpsQA": {}}

    db = SQLiteDbAdapter.load(tmp_path)
    try:
        first = sync_note_types(anki, fs, tmp_path, db)
        second = sync_note_types(anki, fs, tmp_path, db)
    finally:
        db.close()

    assert first == "1 synced"
    assert second == "1 synced"
    assert anki.fetch_model_states.call_count == 2


def test_sync_note_types_without_db_does_not_cache(tmp_path):
    fs = MagicMock()
    fs.load_note_type_configs.return_value = [_qa_config(css=".card { color: black; }")]

    anki = MagicMock()
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    anki.fetch_model_states.return_value = {"AnkiOpsQA": {}}

    sync_note_types(anki, fs, tmp_path, None)
    sync_note_types(anki, fs, tmp_path, None)

    assert anki.fetch_model_states.call_count == 2

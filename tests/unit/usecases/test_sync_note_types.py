"""Tests for note type sync behavior and cache usage."""

from __future__ import annotations

from ankiops.db import SQLiteDbAdapter
from ankiops.models import Field, NoteTypeConfig
from ankiops.sync_note_types import sync_note_types


def _qa_config(*, css: str) -> NoteTypeConfig:
    return NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[
            Field(name="Question", label="Q:", identifying=True),
            Field(name="Answer", label="A:", identifying=True),
        ],
        css=css,
        templates=[{"Name": "Card 1", "Front": "{{Question}}", "Back": "{{Answer}}"}],
    )


class _FakeFs:
    def __init__(self, *config_sets):
        self._config_sets = list(config_sets)
        self.load_count = 0

    def load_note_type_configs(self, _note_types_dir):
        self.load_count += 1
        if len(self._config_sets) == 1:
            return self._config_sets[0]
        return self._config_sets.pop(0)


class _FakeAnkiModels:
    def __init__(self):
        self.state_fetches: list[list[str]] = []
        self.updated: list[list[str]] = []
        self.created: list[list[str]] = []

    def fetch_model_names(self):
        return ["AnkiOpsQA"]

    def fetch_model_states(self, model_names):
        self.state_fetches.append(list(model_names))
        return {name: {} for name in model_names}

    def update_models(self, models, _states):
        self.updated.append([model.name for model in models])

    def create_models(self, models):
        self.created.append([model.name for model in models])


def test_sync_note_types_uses_cache_when_unchanged(tmp_path):
    fs = _FakeFs([_qa_config(css=".card { color: black; }")])
    anki = _FakeAnkiModels()

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        first = sync_note_types(anki, fs, tmp_path, db)
        assert first == "1 synced"
        assert anki.state_fetches == [["AnkiOpsQA"]]
        assert anki.updated == [["AnkiOpsQA"]]

        anki.state_fetches.clear()
        anki.updated.clear()

        second = sync_note_types(anki, fs, tmp_path, db)
        assert second == "1 up to date (cached)"
        assert anki.state_fetches == []
        assert anki.updated == []
    finally:
        db.close()


def test_sync_note_types_cache_invalidates_on_local_change(tmp_path):
    fs = _FakeFs(
        [_qa_config(css=".card { color: black; }")],
        [_qa_config(css=".card { color: blue; }")],
    )
    anki = _FakeAnkiModels()

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        first = sync_note_types(anki, fs, tmp_path, db)
        second = sync_note_types(anki, fs, tmp_path, db)
    finally:
        db.close()

    assert first == "1 synced"
    assert second == "1 synced"
    assert anki.state_fetches == [["AnkiOpsQA"], ["AnkiOpsQA"]]


def test_sync_note_types_without_db_does_not_cache(tmp_path):
    fs = _FakeFs([_qa_config(css=".card { color: black; }")])
    anki = _FakeAnkiModels()

    sync_note_types(anki, fs, tmp_path, None)
    sync_note_types(anki, fs, tmp_path, None)

    assert anki.state_fetches == [["AnkiOpsQA"], ["AnkiOpsQA"]]

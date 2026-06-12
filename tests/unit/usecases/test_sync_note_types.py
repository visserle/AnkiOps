"""Tests for note type sync behavior and cache usage."""

from __future__ import annotations

from ankiops.note_types import NoteField, NoteType, sync_note_type_configs
from ankiops.sync.state import SyncState


def _qa_config(*, css: str) -> NoteType:
    return NoteType(
        name="AnkiOpsQA",
        fields=[
            NoteField(name="Question", label="Q:", identifying=True),
            NoteField(name="Answer", label="A:", identifying=True),
        ],
        css=css,
        templates=[{"Name": "Card 1", "Front": "{{Question}}", "Back": "{{Answer}}"}],
    )


class _FakeAnkiModels:
    def __init__(self):
        self.state_fetches: list[list[str]] = []
        self.updated: list[list[str]] = []
        self.created: list[list[str]] = []

    def fetch_note_type_names(self):
        return ["AnkiOpsQA"]

    def fetch_note_type_states(self, note_type_names):
        self.state_fetches.append(list(note_type_names))
        return {name: {} for name in note_type_names}

    def update_note_types(self, note_types, _states):
        self.updated.append([note_type.name for note_type in note_types])

    def create_note_types(self, note_types):
        self.created.append([note_type.name for note_type in note_types])


def test_sync_note_types_uses_cache_when_unchanged(tmp_path):
    configs = [_qa_config(css=".card { color: black; }")]
    anki = _FakeAnkiModels()

    db = SyncState.open(tmp_path)
    try:
        first = sync_note_type_configs(anki, configs, sync_state=db)
        assert first == "1 synced"
        assert anki.state_fetches == [["AnkiOpsQA"]]
        assert anki.updated == [["AnkiOpsQA"]]

        anki.state_fetches.clear()
        anki.updated.clear()

        second = sync_note_type_configs(anki, configs, sync_state=db)
        assert second == "1 up to date (cached)"
        assert anki.state_fetches == []
        assert anki.updated == []
    finally:
        db.close()


def test_sync_note_types_cache_invalidates_on_local_change(tmp_path):
    first_configs = [_qa_config(css=".card { color: black; }")]
    second_configs = [_qa_config(css=".card { color: blue; }")]
    anki = _FakeAnkiModels()

    db = SyncState.open(tmp_path)
    try:
        first = sync_note_type_configs(anki, first_configs, sync_state=db)
        second = sync_note_type_configs(anki, second_configs, sync_state=db)
    finally:
        db.close()

    assert first == "1 synced"
    assert second == "1 synced"
    assert anki.state_fetches == [["AnkiOpsQA"], ["AnkiOpsQA"]]


def test_sync_note_types_without_db_does_not_cache(tmp_path):
    configs = [_qa_config(css=".card { color: black; }")]
    anki = _FakeAnkiModels()

    sync_note_type_configs(anki, configs, sync_state=None)
    sync_note_type_configs(anki, configs, sync_state=None)

    assert anki.state_fetches == [["AnkiOpsQA"], ["AnkiOpsQA"]]

"""Unit tests for internal import/export sync helper branches."""

from __future__ import annotations

from pathlib import Path

from ankiops.config import get_note_types_dir
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import _sync_deck
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import _PendingWrite, _flush_writes
from ankiops.models import AnkiNote, MarkdownFile, Note
from tests.support.assertions import assert_summary


def _mk_markdown_state(
    path: Path, raw: str, notes: list[Note] | None = None
) -> MarkdownFile:
    return MarkdownFile(file_path=path, raw_content=raw, notes=notes or [])


def test_flush_writes_handles_duplicate_first_lines_with_explicit_note_mapping(
    tmp_path,
):
    fs_port = FileSystemAdapter()
    md_path = tmp_path / "DuplicateLines.md"
    raw = "Q: Same\nA: One\n\n---\n\nQ: Same\nA: Two\n"
    md_path.write_text(raw, encoding="utf-8")

    first = Note(
        note_key=None,
        note_type="AnkiOpsQA",
        fields={"Question": "Same", "Answer": "One"},
    )
    second = Note(
        note_key=None,
        note_type="AnkiOpsQA",
        fields={"Question": "Same", "Answer": "Two"},
    )
    pending = _PendingWrite(
        file_state=_mk_markdown_state(md_path, raw, notes=[first, second]),
        note_key_assignments=[(first, "k1"), (second, "k2")],
    )

    _flush_writes(fs_port, [pending])

    content = md_path.read_text(encoding="utf-8")
    assert "<!-- note_key: k1 -->" in content
    assert "<!-- note_key: k2 -->" in content


def test_flush_writes_updates_existing_key_comment(tmp_path):
    fs_port = FileSystemAdapter()
    md_path = tmp_path / "UpdateKey.md"
    raw = "<!-- note_key: old-key -->\nQ: Q1\nA: A1\n"
    md_path.write_text(raw, encoding="utf-8")

    note = Note(
        note_key="old-key",
        note_type="AnkiOpsQA",
        fields={"Question": "Q1", "Answer": "A1"},
    )
    pending = _PendingWrite(
        file_state=_mk_markdown_state(md_path, raw, notes=[note]),
        note_key_assignments=[(note, "new-key")],
    )

    _flush_writes(fs_port, [pending])

    content = md_path.read_text(encoding="utf-8")
    assert "<!-- note_key: new-key -->" in content
    assert "<!-- note_key: old-key -->" not in content


def test_sync_deck_records_unknown_note_type_error(tmp_path):
    fs_port = FileSystemAdapter()
    configs = fs_port.load_note_type_configs(get_note_types_dir())
    db = SQLiteDbAdapter.load(tmp_path)
    try:
        unknown = AnkiNote(
            note_id=123,
            note_type="UnknownModel",
            fields={"Question": "Q", "Answer": "A"},
            card_ids=[999],
        )
        result = _sync_deck(
            deck_name="UnknownDeck",
            deck_id=10,
            anki_notes=[unknown],
            config_by_name={config.name: config for config in configs},
            existing_file_path=None,
            collection_dir=tmp_path,
            fs_port=fs_port,
            db_port=db,
            note_keys_by_id={},
            pending_note_mappings=[],
            note_fingerprints_by_note_key={},
            pending_fingerprints=[],
        )

        assert len(result.errors) == 1
        assert "Unknown note type UnknownModel" in result.errors[0]
        assert_summary(result.summary, total=0)
    finally:
        db.close()

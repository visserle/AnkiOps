"""Focused tests for sync write gates that protect user Markdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import _can_skip_markdown_rebuild, _sync_deck
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import _flush_writes, _PendingWrite
from ankiops.markdown_format import NOTE_SEPARATOR
from ankiops.models import (
    AnkiNote,
    Change,
    ChangeType,
    Field,
    MarkdownFile,
    Note,
    NoteTypeConfig,
    SyncResult,
)


def _qa_config() -> NoteTypeConfig:
    return NoteTypeConfig(
        name="AnkiOpsQA",
        fields=[
            Field(name="Question", label="Q:", identifying=True),
            Field(name="Answer", label="A:", identifying=True),
        ],
    )


def _configured_fs() -> FileSystemAdapter:
    fs = FileSystemAdapter()
    fs.set_configs([_qa_config()])
    return fs


def _note(
    question: str,
    answer: str,
    *,
    note_key: str | None = None,
) -> Note:
    return Note(
        note_key=note_key,
        note_type="AnkiOpsQA",
        fields={"Question": question, "Answer": answer},
    )


def _raw_note(
    question: str,
    answer: str,
    *,
    note_key: str | None = None,
    note_type: str | None = "AnkiOpsQA",
) -> str:
    lines = []
    if note_key is not None:
        lines.append(f"<!-- note_key: {note_key} -->")
    if note_type is not None:
        lines.append(f"<!-- note_type: {note_type} -->")
    lines.extend([f"Q: {question}", f"A: {answer}"])
    return "\n".join(lines) + "\n"


class _WriteRecorder:
    def __init__(self) -> None:
        self.writes: list[tuple[Path, str]] = []

    def write_markdown_file(self, file_path: Path, content: str) -> None:
        self.writes.append((file_path, content))


def test_flush_writes_persists_assigned_note_key_and_note_type_metadata():
    note = _note("Question", "Answer")
    file_state = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content="Q: Question\nA: Answer\n",
        notes=[note],
    )
    fs = _WriteRecorder()

    _flush_writes(fs, [_PendingWrite(file_state, [(note, "generated-key")])])

    assert fs.writes == [
        (
            Path("Deck.md"),
            (
                "<!-- note_key: generated-key -->\n"
                "<!-- note_type: AnkiOpsQA -->\n"
                "Q: Question\n"
                "A: Answer\n"
            ),
        )
    ]


def test_flush_writes_does_not_touch_current_metadata():
    note = _note("Question", "Answer", note_key="existing-key")
    raw_content = _raw_note("Question", "Answer", note_key="existing-key")
    file_state = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content=raw_content,
        notes=[note],
    )
    fs = _WriteRecorder()

    _flush_writes(fs, [_PendingWrite(file_state, [])])

    assert fs.writes == []


def test_flush_writes_fails_when_note_blocks_and_parsed_notes_diverge():
    file_state = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content="Q: One\nA: One\n",
        notes=[
            _note("One", "One"),
            _note("Two", "Two"),
        ],
    )

    with pytest.raises(ValueError, match="Failed to align parsed notes"):
        _flush_writes(_WriteRecorder(), [_PendingWrite(file_state, [])])


def test_export_skip_gate_requires_current_note_type_metadata():
    note = _note("Question", "Answer", note_key="stable-key")
    fs = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content=_raw_note(
            "Question",
            "Answer",
            note_key="stable-key",
            note_type=None,
        ),
        notes=[note],
    )
    result = SyncResult.for_notes(name="Deck", file_path=fs.file_path)
    result.add_change(Change(ChangeType.SKIP, 101, note.identifier))

    assert not _can_skip_markdown_rebuild(
        is_first_export=False,
        fs=fs,
        final_notes=[note],
        result=result,
        resolve_errors=[],
    )


def test_export_skip_gate_requires_same_note_objects():
    existing = _note("Question", "Answer", note_key="stable-key")
    replacement = _note("Question", "Answer", note_key="stable-key")
    fs = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content=_raw_note("Question", "Answer", note_key="stable-key"),
        notes=[existing],
    )
    result = SyncResult.for_notes(name="Deck", file_path=fs.file_path)
    result.add_change(Change(ChangeType.SKIP, 101, existing.identifier))

    assert not _can_skip_markdown_rebuild(
        is_first_export=False,
        fs=fs,
        final_notes=[replacement],
        result=result,
        resolve_errors=[],
    )


def test_export_skip_gate_allows_true_noop_with_current_metadata():
    note = _note("Question", "Answer", note_key="stable-key")
    fs = MarkdownFile(
        file_path=Path("Deck.md"),
        raw_content=_raw_note("Question", "Answer", note_key="stable-key"),
        notes=[note],
    )
    result = SyncResult.for_notes(name="Deck", file_path=fs.file_path)
    result.add_change(Change(ChangeType.SKIP, 101, note.identifier))

    assert _can_skip_markdown_rebuild(
        is_first_export=False,
        fs=fs,
        final_notes=[note],
        result=result,
        resolve_errors=[],
    )


def test_sync_deck_writes_markdown_when_anki_note_changes(tmp_path):
    deck_path = tmp_path / "Deck.md"
    deck_path.write_text(
        _raw_note("Original", "Old answer", note_key="stable-key"),
        encoding="utf-8",
    )
    anki_note = AnkiNote(
        note_id=101,
        note_type="AnkiOpsQA",
        fields={
            "Question": "Original",
            "Answer": "New answer",
            "AnkiOps Key": "stable-key",
        },
        card_ids=[1001],
    )
    pending_fingerprints: list[tuple[str, str, str]] = []

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        result = _sync_deck(
            "Deck",
            [anki_note],
            {"AnkiOpsQA": _qa_config()},
            deck_path,
            tmp_path,
            _configured_fs(),
            db,
            {101: "stable-key"},
            [],
            {},
            pending_fingerprints,
        )
    finally:
        db.close()

    assert "A: New answer" in deck_path.read_text(encoding="utf-8")
    assert result.summary.updated == 1
    assert pending_fingerprints


def test_sync_deck_preserves_keyless_local_notes_during_export(tmp_path):
    deck_path = tmp_path / "Deck.md"
    deck_path.write_text(
        NOTE_SEPARATOR.join(
            [
                _raw_note("Remote", "Answer", note_key="stable-key").rstrip(),
                _raw_note("Local draft", "Keep me", note_type=None).rstrip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    anki_note = AnkiNote(
        note_id=101,
        note_type="AnkiOpsQA",
        fields={
            "Question": "Remote",
            "Answer": "Answer",
            "AnkiOps Key": "stable-key",
        },
        card_ids=[1001],
    )

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        result = _sync_deck(
            "Deck",
            [anki_note],
            {"AnkiOpsQA": _qa_config()},
            deck_path,
            tmp_path,
            _configured_fs(),
            db,
            {101: "stable-key"},
            [],
            {},
            [],
        )
    finally:
        db.close()

    output = deck_path.read_text(encoding="utf-8")
    assert "Q: Local draft" in output
    assert "A: Keep me" in output
    assert result.protected_keyless_notes == 1

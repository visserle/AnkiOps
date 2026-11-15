"""Field label matching in Markdown deck files."""

from pathlib import Path

import pytest

from ankiops.markdown import read_deck_file
from ankiops.note_types import NoteField, NoteType


@pytest.mark.parametrize("question_label", ["Q:", "Question:"])
def test_field_labels_keep_multiline_content_and_code_fences(
    tmp_path: Path, question_label: str
) -> None:
    note_type = NoteType(
        name="Test",
        fields=[
            NoteField("Question", question_label, True),
            NoteField("Answer", "A:", True),
        ],
    )
    path = tmp_path / "deck.md"
    path.write_text(
        f"{question_label} First line\n"
        "Continuation\n"
        f"{question_label}\tNot a field\n"
        "A:\n"
        "```\n"
        f"{question_label} not a field\n"
        "```\n"
        "Last line",
        encoding="utf-8",
    )

    note = read_deck_file(path, note_types=[note_type]).notes[0]

    assert note.fields == {
        "Question": f"First line\nContinuation\n{question_label}\tNot a field",
        "Answer": f"```\n{question_label} not a field\n```\nLast line",
    }


@pytest.mark.parametrize("label", ["Question text:", " Q:", "Q: "])
def test_field_labels_cannot_contain_spaces(label: str) -> None:
    with pytest.raises(ValueError, match="Field 'Question'.*must not contain spaces"):
        NoteField("Question", label, True)


@pytest.mark.parametrize("label", ["Q:", "Question:"])
def test_duplicate_label_still_reports_the_line(tmp_path: Path, label: str) -> None:
    note_type = NoteType(name="Test", fields=[NoteField("Question", label, True)])
    path = tmp_path / "deck.md"
    path.write_text(f"{label} First\n{label} Again", encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Duplicate field .* \(file: deck\.md, line: 2\)"
    ):
        read_deck_file(path, note_types=[note_type])


def test_empty_label_matches_leading_space_as_before(tmp_path: Path) -> None:
    note_type = NoteType(
        name="Test",
        fields=[NoteField("First", "F:", True), NoteField("Blank", "", False)],
    )
    path = tmp_path / "deck.md"
    path.write_text("F: first\n second", encoding="utf-8")

    note = read_deck_file(path, note_types=[note_type]).notes[0]

    assert note.fields == {"First": "first", "Blank": "second"}

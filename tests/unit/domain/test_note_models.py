"""Tests for note type parsing, validation, and inference."""

import pytest

from ankiops.models import Note


class TestParseChoiceBlock:
    """Test FileSystemAdapter.read_markdown_file with AnkiOpsChoice blocks."""

    def test_choice_with_note_id(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text(
            "<!-- note_key: key-1 -->\n"
            "Q: What is the capital?\n"
            "C1: Berlin\n"
            "C2: Paris\n"
            "A: 1"
        )
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_key == "key-1"
        assert note.note_type == "AnkiOpsChoice"
        assert note.fields["Question"] == "What is the capital?"
        assert note.fields["Choice 1"] == "Berlin"
        assert note.fields["Choice 2"] == "Paris"
        assert note.fields["Answer"] == "1"

    def test_choice_without_id_detected_from_prefix(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("Q: What?\nC1: A\nC2: B\nA: 2")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_type == "AnkiOpsChoice"

    def test_choice_with_all_fields(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text(
            "Q: What?\n"
            "C1: A\nC2: B\nC3: C\nC4: D\nC5: E\nC6: F\nC7: G\n"
            "A: 3\n"
            "E: Extra info"
        )
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_type == "AnkiOpsChoice"
        assert (
            len(
                [
                    field_name
                    for field_name in note.fields
                    if field_name.startswith("Choice")
                ]
            )
            == 7
        )


class TestValidateChoiceNote:
    """Test Note.validate() for AnkiOpsChoice notes."""

    def test_valid_single_choice(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "1"},
        )
        assert note.validate(choice_config) == []

    def test_valid_multiple_choice(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Answer": "1, 2",
            },
        )
        assert note.validate(choice_config) == []

    def test_missing_mandatory_question(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Choice 1": "A", "Choice 2": "B", "Answer": "1"},
        )
        errors = note.validate(choice_config)
        assert any(
            "missing mandatory field" in error.lower() or "Missing mandatory" in error
            for error in errors
        )

    def test_missing_mandatory_choice(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Answer": "1"},
        )
        errors = note.validate(choice_config)
        assert any("at least 2 choices" in error for error in errors)

    def test_only_one_choice(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Answer": "1"},
        )
        errors = note.validate(choice_config)
        assert any("at least 2 choices" in error for error in errors)

    def test_missing_mandatory_answer(self, choice_config):
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B"},
        )
        errors = note.validate(choice_config)
        assert any(
            "missing mandatory field" in error.lower() or "Missing mandatory" in error
            for error in errors
        )

    def test_answer_out_of_range_too_high(self, choice_config):
        """Answer references choice number that doesn't exist."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "3"},
        )
        errors = note.validate(choice_config)
        assert any("3" in error and "2 choice" in error for error in errors)

    def test_answer_out_of_range_zero(self, choice_config):
        """Answer with 0 is invalid (choices start at 1)."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "0"},
        )
        errors = note.validate(choice_config)
        assert any("0" in error for error in errors)

    # -- Restored edge cases (dropped during Hexagonal Architecture refactor) --

    def test_invalid_answer_not_integer(self, choice_config):
        """Non-integer answer should produce validation error."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "abc"},
        )
        errors = note.validate(choice_config)
        assert any("integer" in error.lower() for error in errors)

    def test_invalid_answer_mixed_content(self, choice_config):
        """Mixed content in multi-answer should produce validation error."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Answer": "1, abc, 2",
            },
        )
        errors = note.validate(choice_config)
        assert any("integer" in error.lower() for error in errors)

    def test_answer_out_of_range_negative(self, choice_config):
        """Negative answer is invalid."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "-1"},
        )
        errors = note.validate(choice_config)
        assert len(errors) > 0

    def test_valid_with_seven_choices(self, choice_config):
        """All 7 choices can be used."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Choice 3": "C",
                "Choice 4": "D",
                "Choice 5": "E",
                "Choice 6": "F",
                "Choice 7": "G",
                "Answer": "7",
            },
        )
        assert note.validate(choice_config) == []

    def test_answer_exceeds_seven(self, choice_config):
        """Answer > 7 is always invalid."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q", "Choice 1": "A", "Choice 2": "B", "Answer": "8"},
        )
        errors = note.validate(choice_config)
        assert len(errors) > 0

    def test_valid_sparse_choices(self, choice_config):
        """Choices don't have to be consecutive (e.g., C1, C2, C4)."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Choice 4": "D",
                "Answer": "4",
            },
        )
        errors = note.validate(choice_config)
        assert errors == []

    def test_multiple_answers_one_out_of_range(self, choice_config):
        """If any answer in multi-choice is out of range, it's an error."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Choice 3": "C",
                "Answer": "1, 2, 5",
            },
        )
        errors = note.validate(choice_config)
        assert any("3 choice" in error for error in errors)

    def test_valid_multiple_choice_three_answers(self, choice_config):
        """Three valid answers in multi-choice."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Choice 3": "C",
                "Choice 4": "D",
                "Answer": "1, 2, 4",
            },
        )
        assert note.validate(choice_config) == []

    def test_answer_with_spaces(self, choice_config):
        """Answer field can have spaces around commas."""
        note = Note(
            note_key=None,
            note_type="AnkiOpsChoice",
            fields={
                "Question": "Q",
                "Choice 1": "A",
                "Choice 2": "B",
                "Choice 3": "C",
                "Answer": "1 , 2 ,3",
            },
        )
        assert note.validate(choice_config) == []


class TestParseChoiceEdgeCases:
    """Test parsing edge cases for choice notes."""

    def test_choice_multiline_question(self, fs, tmp_path):
        """Multiline questions should be parsed correctly."""
        md = tmp_path / "deck.md"
        md.write_text("Q: First line\nSecond line\nC1: A\nC2: B\nA: 1")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert "First line" in note.fields["Question"]
        assert "Second line" in note.fields["Question"]


class TestParseClozeBlock:
    """Test FileSystemAdapter.read_markdown_file with AnkiOpsCloze blocks."""

    def test_cloze_with_note_id(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("<!-- note_key: key-1 -->\nT: This is a {{c1::cloze}} deletion")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_key == "key-1"
        assert note.note_type == "AnkiOpsCloze"
        assert "{{c1::cloze}}" in note.fields["Text"]

    def test_cloze_without_id_detected_from_prefix(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("T: This is {{c1::a test}}")
        result = fs.read_markdown_file(md)
        assert result.notes[0].note_type == "AnkiOpsCloze"

    def test_cloze_with_hint(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("T: The {{c1::sun::star}} is bright")
        result = fs.read_markdown_file(md)
        assert "{{c1::sun::star}}" in result.notes[0].fields["Text"]

    def test_cloze_all_fields(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("T: {{c1::text}}\nE: extra info")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.fields["Text"] == "{{c1::text}}"
        assert note.fields["Extra"] == "extra info"


class TestParseQABlock:
    """Test FileSystemAdapter.read_markdown_file with AnkiOpsQA blocks."""

    def test_qa_with_note_id(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("<!-- note_key: key-1 -->\nQ: What is 2+2?\nA: 4")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_key == "key-1"
        assert note.note_type == "AnkiOpsQA"
        assert note.fields["Question"] == "What is 2+2?"
        assert note.fields["Answer"] == "4"

    def test_qa_without_id(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("Q: What?\nA: Answer")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.note_key is None
        assert note.note_type == "AnkiOpsQA"

    def test_qa_with_extra(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("Q: Question\nA: Answer\nE: Extra notes")
        result = fs.read_markdown_file(md)
        note = result.notes[0]
        assert note.fields["Extra"] == "Extra notes"


class TestMultiNoteFile:
    """Test parsing files with multiple notes separated by ---."""

    def test_two_notes(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("Q: Q1\nA: A1\n\n---\n\nQ: Q2\nA: A2")
        result = fs.read_markdown_file(md)
        assert len(result.notes) == 2
        assert result.notes[0].fields["Question"] == "Q1"
        assert result.notes[1].fields["Question"] == "Q2"


class TestNoteInference:
    """Test note type inference from field names/prefixes."""

    def test_infer_qa(self, fs):
        assert fs._infer_note_type({"Question": "q", "Answer": "a"}) == "AnkiOpsQA"

    def test_infer_cloze(self, fs):
        assert fs._infer_note_type({"Text": "t"}, prefixes={"T:"}) == "AnkiOpsCloze"

    def test_infer_text_without_prefix_is_ambiguous(self, fs):
        with pytest.raises(ValueError, match="Ambiguous note type"):
            fs._infer_note_type({"Text": "t"})

    def test_infer_choice(self, fs):
        fields = {"Question": "q", "Answer": "a", "Choice 1": "c1", "Choice 2": "c2"}
        assert fs._infer_note_type(fields) == "AnkiOpsChoice"

    def test_infer_input(self, fs):
        # AnkiOpsInput requires Question + Input (both identifying)
        fields = {"Question": "q", "Input": "i"}
        assert fs._infer_note_type(fields) == "AnkiOpsInput"

    def test_unknown_fails(self, fs):
        with pytest.raises(ValueError, match="Cannot determine note type"):
            fs._infer_note_type({"Unknown": "v", "Random": "r"})

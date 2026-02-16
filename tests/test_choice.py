"""Tests for AnkiOpsChoice note type parsing and validation."""

import pytest

from ankiops.models import Note


class TestParseChoiceBlock:
    """Test Note.from_block with AnkiOpsChoice blocks."""

    def test_choice_with_note_id(self):
        block = (
            "<!-- note_id: 789 -->\n"
            "Q: What is the capital of France?\n"
            "C1: Paris\n"
            "C2: London\n"
            "C3: Berlin\n"
            "A: 1"
        )
        parsed_note = Note.from_block(block)
        assert parsed_note.note_id == 789
        assert parsed_note.note_type == "AnkiOpsChoice"
        assert parsed_note.fields["Question"] == "What is the capital of France?"
        assert parsed_note.fields["Choice 1"] == "Paris"
        assert parsed_note.fields["Choice 2"] == "London"
        assert parsed_note.fields["Choice 3"] == "Berlin"
        assert parsed_note.fields["Answer"] == "1"

    def test_choice_without_id_detected_from_prefix(self):
        block = "Q: Test\nC1: Choice 1\nC2: Choice 2\nA: 1"
        parsed_note = Note.from_block(block)
        assert parsed_note.note_id is None
        assert parsed_note.note_type == "AnkiOpsChoice"

    def test_choice_with_all_fields(self):
        block = (
            "<!-- note_id: 100 -->\n"
            "Q: Question?\n"
            "C1: A\nC2: B\nC3: C\nC4: D\nC5: E\nC6: F\nC7: G\n"
            "A: 1\n"
            "E: Extra info\n"
            "M: More info"
        )
        parsed_note = Note.from_block(block)
        assert parsed_note.note_type == "AnkiOpsChoice"
        assert parsed_note.fields["Choice 7"] == "G"
        assert parsed_note.fields["Extra"] == "Extra info"
        assert parsed_note.fields["More"] == "More info"

    def test_choice_multiline_question(self):
        block = "Q: First line\nSecond line\nC1: Choice 1\nA: 1"
        parsed_note = Note.from_block(block)
        assert parsed_note.fields["Question"] == "First line\nSecond line"


class TestValidateChoiceNote:
    """Test Note.validate() for AnkiOpsChoice notes."""

    def test_valid_single_choice(self):
        block = "Q: Question?\nC1: Choice 1\nC2: Choice 2\nA: 1"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert errors == []

    def test_valid_multiple_choice(self):
        block = "Q: Question?\nC1: A\nC2: B\nC3: C\nA: 1, 2"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert errors == []

    def test_valid_multiple_choice_three_answers(self):
        block = "Q: Question?\nC1: A\nC2: B\nC3: C\nC4: D\nA: 1, 2, 4"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert errors == []

    def test_answer_with_spaces(self):
        """Answer field can have spaces around commas."""
        block = "Q: Question?\nC1: A\nC2: B\nC3: C\nA: 1 , 2 ,3"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert errors == []

    def test_missing_mandatory_question(self):
        block = "C1: Choice 1\nC2: Choice 2\nA: 1"
        parsed_note = Note(
            note_id=1,
            note_type="AnkiOpsChoice",
            fields={"Choice 1": "A", "Choice 2": "B", "Answer": "1"},
        )
        errors = parsed_note.validate()
        assert any("Question" in e for e in errors)

    def test_missing_mandatory_choice(self):
        parsed_note = Note(
            note_id=1,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q?", "Answer": "1"},
        )
        errors = parsed_note.validate()
        assert any("at least 2 choices" in e for e in errors)

    def test_only_one_choice(self):
        parsed_note = Note(
            note_id=1,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q?", "Choice 1": "A", "Answer": "1"},
        )
        errors = parsed_note.validate()
        assert any("at least 2 choices" in e for e in errors)

    def test_missing_mandatory_answer(self):
        block = "Q: Question?\nC1: Choice 1\nC2: Choice 2"
        parsed_note = Note(
            note_id=1,
            note_type="AnkiOpsChoice",
            fields={"Question": "Q?", "Choice 1": "A", "Choice 2": "B"},
        )
        errors = parsed_note.validate()
        assert any("Answer" in e or "A:" in e for e in errors)

    def test_invalid_answer_not_integer(self):
        block = "Q: Question?\nC1: A\nC2: B\nA: abc"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert any("integers" in e for e in errors)

    def test_invalid_answer_mixed_content(self):
        block = "Q: Question?\nC1: A\nC2: B\nA: 1, abc, 2"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert any("integers" in e for e in errors)

    def test_answer_out_of_range_too_high(self):
        """Answer references choice number that doesn't exist."""
        block = "Q: Question?\nC1: A\nC2: B\nA: 3"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert any("only 2 choice(s) are provided" in e for e in errors)

    def test_answer_out_of_range_zero(self):
        """Answer with 0 is invalid (choices start at 1)."""
        block = "Q: Question?\nC1: A\nC2: B\nA: 0"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert len(errors) > 0

    def test_answer_out_of_range_negative(self):
        """Negative answer is invalid."""
        block = "Q: Question?\nC1: A\nC2: B\nA: -1"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert len(errors) > 0

    def test_valid_with_seven_choices(self):
        """All 7 choices can be used."""
        block = "Q: Question?\nC1: A\nC2: B\nC3: C\nC4: D\nC5: E\nC6: F\nC7: G\nA: 7"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert errors == []

    def test_answer_exceeds_seven(self):
        """Answer > 7 is always invalid."""
        block = "Q: Question?\nC1: A\nC2: B\nA: 8"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert len(errors) > 0

    def test_valid_sparse_choices(self):
        """Choices don't have to be consecutive (e.g., C1, C2, C4)."""
        block = "Q: Question?\nC1: A\nC2: B\nC4: D\nA: 4"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        # Should be valid - answer 4 is provided
        assert errors == []

    def test_multiple_answers_one_out_of_range(self):
        """If any answer in multi-choice is out of range, it's an error."""
        block = "Q: Question?\nC1: A\nC2: B\nC3: C\nA: 1, 2, 5"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert any("only 3 choice(s) are provided" in e for e in errors)

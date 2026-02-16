"""Tests for note parsing, formatting, and round-trip (both QA and Cloze)."""

import pytest

from ankiops.config import NOTE_SEPARATOR
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import AnkiNote, FileState, Note


class TestParseClozBlock:
    """Test Note.from_block with AnkiOpsCloze blocks."""

    def test_cloze_with_note_id(self):
        block = (
            "<!-- note_id: 789 -->\n"
            "T: The capital of {{c1::France}} is {{c2::Paris}}\n"
            "E: Geography fact"
        )
        parsed_note = Note.from_block(block)
        assert parsed_note.note_id == 789
        assert parsed_note.note_type == "AnkiOpsCloze"
        assert len(parsed_note.fields) == 2
        assert (
            parsed_note.fields["Text"]
            == "The capital of {{c1::France}} is {{c2::Paris}}"
        )
        assert parsed_note.fields["Extra"] == "Geography fact"

    def test_cloze_without_id_detected_from_prefix(self):
        block = "T: This is a {{c1::cloze}} test\nE: Extra"
        parsed_note = Note.from_block(block)
        assert parsed_note.note_id is None
        assert parsed_note.note_type == "AnkiOpsCloze"
        assert parsed_note.fields["Text"] == "This is a {{c1::cloze}} test"

    def test_cloze_with_hint(self):
        block = (
            "<!-- note_id: 100 -->\n"
            "T: The {{c1::mitochondria::organelle}} is the powerhouse of the cell"
        )
        parsed_note = Note.from_block(block)
        assert parsed_note.note_type == "AnkiOpsCloze"
        assert "{{c1::mitochondria::organelle}}" in parsed_note.fields["Text"]

    def test_cloze_multiline_text(self):
        block = (
            "<!-- note_id: 200 -->\n"
            "T: First line with {{c1::cloze}}\n"
            "Second line continues\n"
            "E: Some extra info"
        )
        parsed_note = Note.from_block(block)
        assert parsed_note.note_type == "AnkiOpsCloze"
        assert (
            "First line with {{c1::cloze}}\nSecond line continues"
            == parsed_note.fields["Text"]
        )

    def test_cloze_all_fields(self):
        block = "<!-- note_id: 300 -->\nT: {{c1::Answer}}\nE: Extra\nM: More info"
        parsed_note = Note.from_block(block)
        assert parsed_note.fields["Text"] == "{{c1::Answer}}"
        assert parsed_note.fields["Extra"] == "Extra"
        assert parsed_note.fields["More"] == "More info"


class TestParseQABlock:
    """Verify AnkiOpsQA parsing."""

    def test_qa_with_note_id(self):
        block = "<!-- note_id: 123 -->\nQ: What?\nA: This"
        parsed_note = Note.from_block(block)
        assert parsed_note.note_id == 123
        assert parsed_note.note_type == "AnkiOpsQA"
        assert parsed_note.fields["Question"] == "What?"
        assert parsed_note.fields["Answer"] == "This"

    def test_qa_without_id(self):
        block = "Q: New question\nA: New answer"
        parsed_note = Note.from_block(block)
        assert parsed_note.note_type == "AnkiOpsQA"
        assert parsed_note.note_id is None


class TestFormatNote:
    """Test AnkiNote.to_markdown() for both note types."""

    @pytest.fixture
    def converter(self):
        return HTMLToMarkdown()

    def test_format_cloze_note(self, converter):
        anki_note = AnkiNote.from_raw(
            {
                "noteId": 789,
                "modelName": "AnkiOpsCloze",
                "fields": {
                    "Text": {"value": "The {{c1::answer}} is here"},
                    "Extra": {"value": "Extra info"},
                    "More": {"value": ""},
                },
                "cards": [],
            }
        )
        result = anki_note.to_markdown(converter)
        assert "<!-- note_id: 789 -->" in result
        assert "T: The {{c1::answer}} is here" in result
        assert "E: Extra info" in result
        assert "M:" not in result  # empty optional field omitted

    def test_format_qa_card(self, converter):
        anki_note = AnkiNote.from_raw(
            {
                "noteId": 123,
                "modelName": "AnkiOpsQA",
                "fields": {
                    "Question": {"value": "What?"},
                    "Answer": {"value": "This"},
                    "Extra": {"value": ""},
                    "More": {"value": ""},
                },
                "cards": [],
            }
        )
        result = anki_note.to_markdown(converter)
        assert "<!-- note_id: 123 -->" in result
        assert "Q: What?" in result
        assert "A: This" in result


class TestExtractNoteBlocks:
    """Test extract_note_blocks with mixed content."""

    def test_mixed_qa_and_cloze(self):
        content = (
            "<!-- note_id: 10 -->\n"
            "Q: Question\n"
            "A: Answer\n"
            f"{NOTE_SEPARATOR}"
            "<!-- note_id: 20 -->\n"
            "T: {{c1::Cloze}}"
        )
        blocks = FileState.extract_note_blocks(content)
        assert "note_id: 10" in blocks
        assert "note_id: 20" in blocks
        assert len(blocks) == 2

    def test_only_cloze_blocks(self):
        content = (
            "<!-- note_id: 100 -->\n"
            "T: {{c1::First}}\n"
            f"{NOTE_SEPARATOR}"
            "<!-- note_id: 200 -->\n"
            "T: {{c1::Second}}"
        )
        blocks = FileState.extract_note_blocks(content)
        assert "note_id: 100" in blocks
        assert "note_id: 200" in blocks


class TestValidateNote:
    """Test Note.validate() for mandatory fields and unknown prefixes."""

    def test_valid_qa_card(self):
        block = "<!-- note_id: 1 -->\nQ: Question\nA: Answer"
        parsed_note = Note.from_block(block)
        assert parsed_note.validate() == []

    def test_valid_cloze_card(self):
        block = "<!-- note_id: 1 -->\nT: {{c1::text}}"
        parsed_note = Note.from_block(block)
        assert parsed_note.validate() == []

    def test_missing_mandatory_qa_fields(self):
        block = "<!-- note_id: 1 -->\nQ: Question only"
        with pytest.raises(ValueError, match="Cannot determine note type"):
            Note.from_block(block)

    def test_missing_mandatory_cloze_field(self):
        # Construct directly since T: without cloze syntax would fail validation
        parsed_note = Note(
            note_id=1,
            note_type="AnkiOpsCloze",
            fields={"Extra": "Only extra"},
        )
        errors = parsed_note.validate()
        assert any("Text" in e and "T:" in e for e in errors)

    def test_no_unique_prefix_raises(self):
        block = "<!-- note_id: 1 -->\nE: Only extra"
        with pytest.raises(ValueError, match="Cannot determine note type"):
            Note.from_block(block)

    def test_cloze_without_cloze_syntax(self):
        block = "T: This has no cloze deletions"
        parsed_note = Note.from_block(block)
        errors = parsed_note.validate()
        assert any("cloze syntax" in e for e in errors)

    def test_cloze_with_valid_syntax(self):
        block = "T: The {{c1::answer}} is here"
        parsed_note = Note.from_block(block)
        assert parsed_note.validate() == []

    def test_continuation_lines_not_flagged(self):
        block = "<!-- note_id: 1 -->\nQ: Question\nA: Answer starts\nmore answer text"
        parsed_note = Note.from_block(block)
        assert parsed_note.validate() == []


class TestClozeRoundTrip:
    """Test that cloze syntax passes through HTML<->Markdown converters unchanged."""

    @pytest.fixture
    def md_to_html(self):
        return MarkdownToHTML()

    @pytest.fixture
    def html_to_md(self):
        return HTMLToMarkdown()

    def test_cloze_syntax_through_markdown_to_html(self, md_to_html):
        md = "The capital of {{c1::France}} is {{c2::Paris}}"
        html = md_to_html.convert(md)
        assert "{{c1::France}}" in html
        assert "{{c2::Paris}}" in html

    def test_cloze_syntax_through_html_to_markdown(self, html_to_md):
        html = "The capital of {{c1::France}} is {{c2::Paris}}"
        md = html_to_md.convert(html)
        assert "{{c1::France}}" in md
        assert "{{c2::Paris}}" in md

    def test_cloze_with_hint_roundtrip(self, md_to_html, html_to_md):
        original = "The {{c1::mitochondria::organelle}} is the powerhouse"
        html = md_to_html.convert(original)
        md = html_to_md.convert(html)
        assert "{{c1::mitochondria::organelle}}" in md

    def test_cloze_with_formatting(self, md_to_html):
        md = "**Bold** text with {{c1::cloze}} and *italic*"
        html = md_to_html.convert(md)
        assert "{{c1::cloze}}" in html
        assert "<strong>Bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_multiple_clozes_same_number(self, md_to_html):
        md = "{{c1::First}} and {{c1::second}} are both c1"
        html = md_to_html.convert(md)
        assert "{{c1::First}}" in html
        assert "{{c1::second}}" in html

    def test_multiple_clozes_different_numbers(self, md_to_html):
        md = "{{c1::One}}, {{c2::Two}}, {{c3::Three}}, {{c4::Four}}, {{c5::Five}}"
        html = md_to_html.convert(md)
        assert "{{c1::One}}" in html
        assert "{{c2::Two}}" in html
        assert "{{c3::Three}}" in html
        assert "{{c4::Four}}" in html
        assert "{{c5::Five}}" in html

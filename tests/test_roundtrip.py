"""Tests for import/export round-trip scenarios.

Verifies that changes made in markdown are correctly detected when
compared against the Anki-side fields, and that the update payload
sent back to Anki is complete (including cleared optional fields).
"""

import pytest

from ankiops.config import NOTE_CONFIG
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import AnkiNote, Note

# -- helpers -----------------------------------------------------------------


def _anki_note_raw(fields: dict[str, str], note_type: str = "AnkiOpsQA") -> dict:
    """Build a raw AnkiConnect-style note dict from {name: value}."""
    return {
        "noteId": 1,
        "modelName": note_type,
        "fields": {k: {"value": v} for k, v in fields.items()},
        "cards": [],
    }


def _complete_fields(note_type: str, html_fields: dict[str, str]) -> dict[str, str]:
    """Replicate the complete-fields logic from _sync_file."""
    all_field_names = [name for name, _ in NOTE_CONFIG.get(note_type, [])]
    complete = {name: "" for name in all_field_names}
    complete.update(html_fields)
    return complete


def _has_changes(anki_fields: dict[str, str], complete: dict[str, str]) -> bool:
    """Return True if the markdown-derived fields differ from Anki.

    In real Anki, notes always have ALL fields for their note type
    (empty string if unset).  Test fixtures may omit empty fields, so
    a missing key in *anki_fields* is treated as ``""``.
    """
    for k, v in complete.items():
        anki_val = anki_fields.get(k, "")
        if anki_val != v:
            return True
    return False


def _roundtrip(anki_fields: dict[str, str], note_type: str) -> dict[str, str]:
    """Full round-trip: Anki → markdown → parse → html → complete fields."""
    html_to_md = HTMLToMarkdown()
    md_to_html = MarkdownToHTML()

    raw = _anki_note_raw(anki_fields, note_type)
    anki_note = AnkiNote.from_raw(raw)
    md_text = anki_note.to_markdown(html_to_md)
    parsed = Note.from_block(md_text)
    html_fields = parsed.to_html(md_to_html)
    return _complete_fields(note_type, html_fields)


# -- tests: change detection -------------------------------------------------


class TestChangeDetection:
    """The change detection logic must catch additions, removals, and edits."""

    ANKI_QA = {
        "Question": "What is 2+2?",
        "Answer": "4",
        "Extra": "Basic arithmetic",
        "More": "",
    }

    def test_no_change_detected_when_fields_match(self):
        """Identical fields → no change."""
        md_fields = dict(self.ANKI_QA)
        complete = _complete_fields("AnkiOpsQA", md_fields)
        assert not _has_changes(self.ANKI_QA, complete)

    def test_change_detected_when_optional_field_removed(self):
        """Removing an optional field from markdown must be detected."""
        md_fields = {"Question": "What is 2+2?", "Answer": "4"}
        complete = _complete_fields("AnkiOpsQA", md_fields)

        assert complete["Extra"] == ""
        assert _has_changes(self.ANKI_QA, complete)

    def test_change_detected_when_optional_field_added(self):
        """Adding content to a previously-empty optional field."""
        anki = {**self.ANKI_QA, "Extra": "", "More": ""}
        md_fields = {**anki, "Extra": "New extra content"}
        complete = _complete_fields("AnkiOpsQA", md_fields)
        assert _has_changes(anki, complete)

    def test_change_detected_when_value_edited(self):
        """Editing a mandatory field value."""
        md_fields = {**self.ANKI_QA, "Answer": "Four"}
        complete = _complete_fields("AnkiOpsQA", md_fields)
        assert _has_changes(self.ANKI_QA, complete)

    def test_no_change_when_empty_optional_omitted(self):
        """An already-empty optional field missing from markdown is fine."""
        anki = {"Question": "Q?", "Answer": "A", "Extra": "", "More": ""}
        md_fields = {"Question": "Q?", "Answer": "A"}
        complete = _complete_fields("AnkiOpsQA", md_fields)
        assert not _has_changes(anki, complete)

    def test_change_detected_removing_both_optional_fields(self):
        """Removing both Extra and More at once."""
        anki = {"Question": "Q", "Answer": "A", "Extra": "E", "More": "M"}
        md_fields = {"Question": "Q", "Answer": "A"}
        complete = _complete_fields("AnkiOpsQA", md_fields)
        assert complete["Extra"] == ""
        assert complete["More"] == ""
        assert _has_changes(anki, complete)


class TestChangeDetectionCloze:
    """Same scenarios for AnkiOpsCloze notes."""

    def test_removing_extra_from_cloze(self):
        anki = {"Text": "{{c1::Answer}}", "Extra": "Hint", "More": ""}
        md_fields = {"Text": "{{c1::Answer}}"}
        complete = _complete_fields("AnkiOpsCloze", md_fields)
        assert _has_changes(anki, complete)

    def test_no_change_cloze_all_match(self):
        anki = {"Text": "{{c1::Answer}}", "Extra": "Hint", "More": ""}
        md_fields = {"Text": "{{c1::Answer}}", "Extra": "Hint"}
        complete = _complete_fields("AnkiOpsCloze", md_fields)
        assert not _has_changes(anki, complete)


class TestChangeDetectionReversed:
    """Reversed card type."""

    def test_removing_extra_from_reversed(self):
        anki = {"Front": "F", "Back": "B", "Extra": "E", "More": ""}
        md_fields = {"Front": "F", "Back": "B"}
        complete = _complete_fields("AnkiOpsReversed", md_fields)
        assert _has_changes(anki, complete)


class TestChangeDetectionInput:
    """Input card type."""

    def test_removing_extra_from_input(self):
        anki = {"Question": "Q", "Input": "I", "Extra": "E", "More": ""}
        md_fields = {"Question": "Q", "Input": "I"}
        complete = _complete_fields("AnkiOpsInput", md_fields)
        assert _has_changes(anki, complete)


# -- tests: full round-trip --------------------------------------------------


class TestRoundTripWithEdits:
    """Export → edit markdown → re-import should detect the edit."""

    @pytest.fixture
    def md_to_html(self):
        return MarkdownToHTML()

    @pytest.fixture
    def html_to_md(self):
        return HTMLToMarkdown()

    def test_character_conversions(self, md_to_html, html_to_md):
        """Verify arrow and not-equal conversions."""
        test_cases = [
            ("a --> b", "a \u2192 b"),
            ("a ==> b", "a \u21d2 b"),
            ("a =/= b", "a \u2260 b"),
        ]

        for md_in, expected_html in test_cases:
            # Markdown -> HTML
            html_out = md_to_html.convert(md_in)
            assert expected_html in html_out

            # HTML -> Markdown
            md_out = html_to_md.convert(html_out)
            assert md_out == md_in

    def test_character_conversions_complex_roundtrip(self, md_to_html, html_to_md):
        """Verify complex string conversions."""
        md_in = "If a --> b and b ==> c, then a =/= c is maybe false."
        html_out = md_to_html.convert(md_in)

        assert "\u2192" in html_out
        assert "\u21d2" in html_out
        assert "\u2260" in html_out

        md_back = html_to_md.convert(html_out)
        assert md_back == md_in

    def test_remove_extra_field_from_exported_markdown(self, html_to_md, md_to_html):
        """The exact bug scenario: export note with Extra, remove it, re-import."""
        anki_fields = {
            "Question": "What?",
            "Answer": "This",
            "Extra": "Info",
            "More": "",
        }
        raw = _anki_note_raw(anki_fields)
        anki_note = AnkiNote.from_raw(raw)

        # Export
        md_text = anki_note.to_markdown(html_to_md)
        assert "E: Info" in md_text

        # User removes the E: line
        lines = [l for l in md_text.splitlines() if not l.startswith("E:")]
        edited_md = "\n".join(lines)
        assert "E:" not in edited_md

        # Re-import
        parsed = Note.from_block(edited_md)
        assert "Extra" not in parsed.fields

        html_fields = parsed.to_html(md_to_html)
        complete = _complete_fields("AnkiOpsQA", html_fields)

        assert complete["Extra"] == ""
        assert _has_changes(anki_fields, complete)

    def test_add_extra_field_to_exported_markdown(self, html_to_md, md_to_html):
        """Export note without Extra, add it, re-import."""
        anki_fields = {"Question": "What?", "Answer": "This", "Extra": "", "More": ""}
        raw = _anki_note_raw(anki_fields)
        anki_note = AnkiNote.from_raw(raw)

        md_text = anki_note.to_markdown(html_to_md)
        assert "E:" not in md_text

        # User adds an Extra line
        edited_md = md_text + "\nE: New extra"
        parsed = Note.from_block(edited_md)
        html_fields = parsed.to_html(md_to_html)
        complete = _complete_fields("AnkiOpsQA", html_fields)

        assert "New extra" in complete["Extra"]
        assert _has_changes(anki_fields, complete)

    def test_edit_answer_field(self, html_to_md, md_to_html):
        anki_fields = {"Question": "What?", "Answer": "This", "Extra": "", "More": ""}
        raw = _anki_note_raw(anki_fields)
        anki_note = AnkiNote.from_raw(raw)

        md_text = anki_note.to_markdown(html_to_md)
        edited_md = md_text.replace("A: This", "A: That")

        parsed = Note.from_block(edited_md)
        html_fields = parsed.to_html(md_to_html)
        complete = _complete_fields("AnkiOpsQA", html_fields)

        assert _has_changes(anki_fields, complete)

    def test_no_edit_no_change(self, html_to_md, md_to_html):
        """Unmodified export should not trigger a change."""
        anki = {"Question": "What?", "Answer": "This", "Extra": "Info", "More": ""}
        complete = _roundtrip(anki, "AnkiOpsQA")
        assert not _has_changes(anki, complete)


# -- tests: AnkiNote.from_raw -----------------------------------------------


class TestAnkiNoteFromRaw:
    """Verify the AnkiNote field extraction."""

    def test_extracts_all_fields(self):
        raw = _anki_note_raw({"Question": "Q", "Answer": "A", "Extra": "E", "More": ""})
        anki_note = AnkiNote.from_raw(raw)
        assert anki_note.fields == {
            "Question": "Q",
            "Answer": "A",
            "Extra": "E",
            "More": "",
        }

    def test_empty_fields_preserved(self):
        raw = _anki_note_raw(
            {"Text": "T", "Extra": "", "More": ""}, note_type="AnkiOpsCloze"
        )
        anki_note = AnkiNote.from_raw(raw)
        assert anki_note.fields["Extra"] == ""
        assert anki_note.fields["More"] == ""

"""Tests for sync logic, change detection, and round-trip fidelity.

Includes:
1.  Block reconciliation (listing changes between file and Anki).
2.  Field-level change detection (hashing/comparison).
3.  Round-trip stability (Anki -> Markdown -> Anki).
"""

from pathlib import Path

import pytest

from ankiops.anki_to_markdown import _reconcile_blocks
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import AnkiNote, ChangeType, FileState, Note
from ankiops.note_type_config import registry

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def sample_notes():
    return [
        Note(
            note_key="key-1",
            note_type="AnkiOpsQA",
            fields={"Question": "Q1", "Answer": "A1"},
        ),
        Note(
            note_key=None,
            note_type="AnkiOpsQA",
            fields={"Question": "Q2", "Answer": "A2"},
        ),
        Note(
            note_key="key-2",
            note_type="AnkiOpsQA",
            fields={"Question": "Q3", "Answer": "A3"},
        ),
    ]


@pytest.fixture
def file_state(sample_notes):
    return FileState(
        file_path=Path("dummy.md"),
        raw_content="",
        deck_key="deck-key-101",
        parsed_notes=sample_notes,
    )


# -- Tests: FileState properties ---------------------------------------------


def test_file_state_existing_notes(file_state):
    """FileState.existing_notes should return only notes with Keys."""
    existing = file_state.existing_notes
    assert len(existing) == 2
    assert {n.note_key for n in existing} == {"key-1", "key-2"}


def test_file_state_new_notes(file_state):
    """FileState.new_notes should return only notes without Keys."""
    new = file_state.new_notes
    assert len(new) == 1
    assert new[0].fields["Question"] == "Q2"


# -- Tests: _reconcile_blocks (Sync Logic) -----------------------------------


def test_reconcile_blocks_no_existing():
    """First export: no existing file, should just sort by creation date (note_id)."""
    block_by_id = {
        "note_key: aaaa-2": (2, "Block 2"),
        "note_key: aaaa-1": (1, "Block 1"),
    }
    changes = _reconcile_blocks(block_by_id, existing_blocks={})

    assert len(changes) == 2
    assert changes[0].change_type == ChangeType.CREATE
    assert changes[0].entity_id == 1
    assert changes[0].entity_repr == "note_key: aaaa-1"

    assert changes[1].change_type == ChangeType.CREATE
    assert changes[1].entity_id == 2


def test_reconcile_blocks_with_updates_and_deletes():
    """Existing file: preserve order, update content, detect deletes."""
    existing = {
        "note_key: aaaa-1": "Block 1 (old)",  # Updated
        "note_key: aaaa-2": "Block 2",  # Skipped (no change)
        "note_key: aaaa-3": "Block 3",  # Deleted (not in Anki state)
    }
    block_by_id = {
        "note_key: aaaa-1": (1, "Block 1 (new)"),
        "note_key: aaaa-2": (2, "Block 2"),
        "note_key: aaaa-4": (4, "Block 4"),  # Created
    }

    changes = _reconcile_blocks(block_by_id, existing_blocks=existing)

    # Expected order of processing in _reconcile_blocks:
    # 1. Existing blocks (1, 2, 3)
    # 2. New blocks (4)

    # Change 1: Update note 1
    assert changes[0].change_type == ChangeType.UPDATE
    assert changes[0].entity_id == 1
    assert changes[0].context["block_content"] == "Block 1 (new)"

    # Change 2: Skip note 2
    assert changes[1].change_type == ChangeType.SKIP
    assert changes[1].entity_id == 2

    # Change 3: Delete note 3
    assert changes[2].change_type == ChangeType.DELETE
    assert changes[2].entity_repr == "note_key: aaaa-3"

    assert changes[3].change_type == ChangeType.CREATE
    assert changes[3].entity_id == 4


def test_reconcile_blocks_stale_key_recreation():
    """If a Key is in the file but not in Anki's export, it should be deleted.

    If Anki produces a new Key (via a new note), it maps to CREATE.
    """
    existing = {
        "note_key: old-key-100": "Content A",
    }
    # Anki has a new note with a different Key
    block_by_id = {
        "note_key: new-key-101": (101, "Content A"),
    }

    changes = _reconcile_blocks(block_by_id, existing_blocks=existing)

    # Implementation detail: It detects DELETE old-key-100 and CREATE new-key-101
    assert len(changes) == 2

    ch_del = next(c for c in changes if c.change_type == ChangeType.DELETE)
    assert ch_del.entity_repr == "note_key: old-key-100"

    ch_create = next(c for c in changes if c.change_type == ChangeType.CREATE)
    assert ch_create.entity_id == 101
    assert ch_create.context["block_content"] == "Content A"


def test_reconcile_blocks_content_update_preserves_key():
    """An update to content should generate an UPDATE change with the same Key."""
    existing = {
        "note_key: aaaa-1": "Old Content",
    }
    block_by_id = {
        "note_key: aaaa-1": (1, "New Content"),
    }

    changes = _reconcile_blocks(block_by_id, existing_blocks=existing)

    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.UPDATE
    assert changes[0].entity_id == 1
    assert changes[0].context["block_content"] == "New Content"


# -- Helpers: Change Detection -----------------------------------------------


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
    config = registry.get(note_type)
    all_field_names = [f.name for f in config.fields]
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
    md_text = anki_note.to_markdown(html_to_md, note_key="key-1")
    parsed = Note.from_block(md_text)
    html_fields = parsed.to_html(md_to_html)
    return _complete_fields(note_type, html_fields)


# -- Tests: Change Detection Logic -------------------------------------------


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


# -- Tests: Full Round-Trip --------------------------------------------------


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
        md_text = anki_note.to_markdown(html_to_md, note_key="key-1")
        assert "E: Info" in md_text

        # User removes the E: line
        lines = [line for line in md_text.splitlines() if not line.startswith("E:")]
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

        md_text = anki_note.to_markdown(html_to_md, note_key="key-1")
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

        md_text = anki_note.to_markdown(html_to_md, note_key="key-1")
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


# -- Tests: AnkiNote.from_raw ------------------------------------------------


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

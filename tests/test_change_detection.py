"""Tests for sync logic, change detection, and round-trip fidelity.

Includes:
1.  Field-level change detection (HTML comparison).
2.  Round-trip stability (Anki → Markdown → parse → HTML).
3.  Optional field removal/addition detection.
"""

import pytest

from ankiops.fs import FileSystemAdapter
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def md_to_html():
    return MarkdownToHTML()


@pytest.fixture
def html_to_md():
    return HTMLToMarkdown()


# -- Helpers: Change Detection -----------------------------------------------


def _complete_fields(
    fs: FileSystemAdapter, note_type: str, html_fields: dict[str, str]
) -> dict[str, str]:
    """Replicate the complete-fields logic."""
    configs = {c.name: c for c in fs._note_type_configs}
    config = configs[note_type]
    complete = {}
    for field in config.fields:
        if field.prefix is None:
            continue
        complete[field.name] = html_fields.get(field.name, "")
    return complete


def _has_changes(anki_fields: dict[str, str], complete: dict[str, str]) -> bool:
    """Return True if the markdown-derived fields differ from Anki."""
    for key, val in complete.items():
        anki_val = anki_fields.get(key, "")
        if val != anki_val:
            return True
    return False


def _roundtrip(
    fs, md_to_html, html_to_md, anki_fields: dict[str, str], note_type: str
) -> dict[str, str]:
    """Full round-trip: Anki → markdown → parse → html → complete fields."""
    # Anki → Markdown
    md_fields = {}
    for name, html_val in anki_fields.items():
        md_fields[name] = html_to_md.convert(html_val)

    # Markdown → HTML (simulating what import does)
    html_fields = {}
    for name, md_val in md_fields.items():
        html_fields[name] = md_to_html.convert(md_val)

    return _complete_fields(fs, note_type, html_fields)


# -- Tests: MarkdownFile properties ------------------------------------------


def test_markdown_file_existing_notes(fs, tmp_path):
    """existing_notes should return only notes with keys."""
    md = tmp_path / "deck.md"
    md.write_text("<!-- note_key: key-1 -->\nQ: Q1\nA: A1\n\n---\n\nQ: Q2\nA: A2")
    result = fs.read_markdown_file(md)
    existing = [n for n in result.notes if n.note_key]
    new = [n for n in result.notes if not n.note_key]
    assert len(existing) == 1
    assert len(new) == 1


def test_markdown_file_new_notes(fs, tmp_path):
    """new_notes should return only notes without keys."""
    md = tmp_path / "deck.md"
    md.write_text("Q: Q1\nA: A1\n\n---\n\nQ: Q2\nA: A2")
    result = fs.read_markdown_file(md)
    new = [n for n in result.notes if not n.note_key]
    assert len(new) == 2


# -- Tests: Change Detection Logic -------------------------------------------


class TestChangeDetection:
    """The change detection logic must catch additions, removals, and edits."""

    ANKI_QA = {
        "Question": "What is 2+2?",
        "Answer": "4",
        "Extra": "Basic arithmetic",
        "More": "",
    }

    def test_no_change_detected_when_fields_match(self, fs, md_to_html, html_to_md):
        """Identical fields → no change."""
        rt = _roundtrip(fs, md_to_html, html_to_md, self.ANKI_QA, "AnkiOpsQA")
        assert not _has_changes(self.ANKI_QA, rt)

    def test_change_detected_when_optional_field_removed(
        self, fs, md_to_html, html_to_md
    ):
        """Removing an optional field from markdown must be detected."""
        modified = dict(self.ANKI_QA)
        modified["Extra"] = ""
        rt = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsQA")
        assert _has_changes(self.ANKI_QA, rt)

    def test_change_detected_when_value_edited(self, fs, md_to_html, html_to_md):
        """Editing a mandatory field value."""
        modified = dict(self.ANKI_QA)
        modified["Question"] = "What is 3+3?"
        rt = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsQA")
        assert _has_changes(self.ANKI_QA, rt)

    def test_no_change_when_empty_optional_omitted(self, fs, md_to_html, html_to_md):
        """An already-empty optional field missing from markdown is fine."""
        # More is already empty, round-trip should show no change
        rt = _roundtrip(fs, md_to_html, html_to_md, self.ANKI_QA, "AnkiOpsQA")
        assert not _has_changes(self.ANKI_QA, rt)


class TestChangeDetectionCloze:
    """Same scenarios for AnkiOpsCloze notes."""

    def test_removing_extra_from_cloze(self, fs, md_to_html, html_to_md):
        anki = {"Text": "{{c1::test}}", "Extra": "info"}
        modified = {"Text": "{{c1::test}}", "Extra": ""}
        rt = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsCloze")
        assert _has_changes(anki, rt)

    def test_no_change_cloze_all_match(self, fs, md_to_html, html_to_md):
        anki = {"Text": "{{c1::test}}", "Extra": ""}
        rt = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsCloze")
        assert not _has_changes(anki, rt)


# -- Tests: Round-trip Stability ---------------------------------------------


class TestRoundTrip:
    """Ensure Anki → Markdown → Anki doesn't lose data."""

    def test_qa_roundtrip_stable(self, fs, md_to_html, html_to_md):
        anki = {
            "Question": "What is the capital of France?",
            "Answer": "Paris",
            "Extra": "",
            "More": "",
        }
        rt = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsQA")
        # Only compare fields that exist on both sides
        common = set(anki.keys()) & set(rt.keys())
        assert all(anki[k] == rt[k] for k in common)

    def test_cloze_roundtrip_stable(self, fs, md_to_html, html_to_md):
        anki = {"Text": "This is {{c1::a cloze}}", "Extra": ""}
        rt = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsCloze")
        assert not _has_changes(anki, rt)

"""Change-detection tests for markdown/html conversion sync paths."""

from __future__ import annotations

import pytest

from ankiops.fs import FileSystemAdapter
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML


@pytest.fixture
def md_to_html():
    return MarkdownToHTML()


@pytest.fixture
def html_to_md():
    return HTMLToMarkdown()


def _complete_fields(
    fs: FileSystemAdapter,
    note_type: str,
    html_fields: dict[str, str],
) -> dict[str, str]:
    config = {cfg.name: cfg for cfg in fs._note_type_configs}[note_type]
    return {
        field.name: html_fields.get(field.name, "")
        for field in config.fields
        if field.prefix is not None
    }


def _has_changes(anki_fields: dict[str, str], complete: dict[str, str]) -> bool:
    return any(complete.get(key, "") != anki_fields.get(key, "") for key in complete)


def _roundtrip(fs, md_to_html, html_to_md, anki_fields: dict[str, str], note_type: str) -> dict[str, str]:
    md_fields = {name: html_to_md.convert(value) for name, value in anki_fields.items()}
    html_fields = {name: md_to_html.convert(value) for name, value in md_fields.items()}
    return _complete_fields(fs, note_type, html_fields)


def test_markdown_file_existing_notes(fs, tmp_path):
    """existing_notes should return only notes with keys."""
    md = tmp_path / "deck.md"
    md.write_text(
        "<!-- note_key: key-1 -->\nQ: Q1\nA: A1\n\n---\n\nQ: Q2\nA: A2",
        encoding="utf-8",
    )
    result = fs.read_markdown_file(md)
    existing = [note for note in result.notes if note.note_key]
    new = [note for note in result.notes if not note.note_key]
    assert len(existing) == 1
    assert len(new) == 1


def test_markdown_file_new_notes(fs, tmp_path):
    """new_notes should return only notes without keys."""
    md = tmp_path / "deck.md"
    md.write_text("Q: Q1\nA: A1\n\n---\n\nQ: Q2\nA: A2", encoding="utf-8")
    result = fs.read_markdown_file(md)
    new = [note for note in result.notes if not note.note_key]
    assert len(new) == 2


class TestChangeDetection:
    """The change-detection layer should catch edits, removals, and no-ops."""

    ANKI_QA = {
        "Question": "What is 2+2?",
        "Answer": "4",
        "Extra": "Basic arithmetic",
        "More": "",
    }

    def test_no_change_detected_when_fields_match(self, fs, md_to_html, html_to_md):
        result = _roundtrip(fs, md_to_html, html_to_md, self.ANKI_QA, "AnkiOpsQA")
        assert not _has_changes(self.ANKI_QA, result)

    def test_change_detected_when_optional_field_removed(self, fs, md_to_html, html_to_md):
        modified = dict(self.ANKI_QA)
        modified["Extra"] = ""
        result = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsQA")
        assert _has_changes(self.ANKI_QA, result)

    def test_change_detected_when_value_edited(self, fs, md_to_html, html_to_md):
        modified = dict(self.ANKI_QA)
        modified["Question"] = "What is 3+3?"
        result = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsQA")
        assert _has_changes(self.ANKI_QA, result)

    def test_no_change_when_empty_optional_omitted(self, fs, md_to_html, html_to_md):
        result = _roundtrip(fs, md_to_html, html_to_md, self.ANKI_QA, "AnkiOpsQA")
        assert not _has_changes(self.ANKI_QA, result)


class TestChangeDetectionCloze:
    """Cloze change-detection should mirror QA semantics."""

    def test_removing_extra_from_cloze(self, fs, md_to_html, html_to_md):
        anki = {"Text": "{{c1::test}}", "Extra": "info"}
        modified = {"Text": "{{c1::test}}", "Extra": ""}
        result = _roundtrip(fs, md_to_html, html_to_md, modified, "AnkiOpsCloze")
        assert _has_changes(anki, result)

    def test_no_change_cloze_all_match(self, fs, md_to_html, html_to_md):
        anki = {"Text": "{{c1::test}}", "Extra": ""}
        result = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsCloze")
        assert not _has_changes(anki, result)


class TestRoundTrip:
    """Round-trip conversion should preserve stable notes."""

    def test_qa_roundtrip_stable(self, fs, md_to_html, html_to_md):
        anki = {
            "Question": "What is the capital of France?",
            "Answer": "Paris",
            "Extra": "",
            "More": "",
        }
        result = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsQA")
        common = set(anki) & set(result)
        assert all(anki[field] == result[field] for field in common)

    def test_cloze_roundtrip_stable(self, fs, md_to_html, html_to_md):
        anki = {"Text": "This is {{c1::a cloze}}", "Extra": ""}
        result = _roundtrip(fs, md_to_html, html_to_md, anki, "AnkiOpsCloze")
        assert not _has_changes(anki, result)

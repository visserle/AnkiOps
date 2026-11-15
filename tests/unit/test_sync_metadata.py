"""Metadata normalization must preserve placement and fenced content."""

from ankiops.notes import Note
from ankiops.sync.to_anki_deck import _upsert_note_metadata_in_block


def test_metadata_is_moved_before_tags_even_if_already_at_start() -> None:
    note = Note(note_key="key", note_type="QA", fields={"Question": "Hello"})
    block = (
        "<!-- note_key: key -->\n<!-- note_type: QA -->\n"
        "Q: Hello\n<!-- tags: example -->\nA: Answer"
    )
    assert _upsert_note_metadata_in_block(block, note, "key") == (
        "Q: Hello\n<!-- note_key: key -->\n<!-- note_type: QA -->\n"
        "<!-- tags: example -->\nA: Answer"
    )


def test_duplicate_metadata_is_removed_but_fenced_comments_are_preserved() -> None:
    note = Note(note_key="key", note_type="QA", fields={"Question": "Hello"})
    block = (
        "<!-- note_key: stale -->\n<!-- note_type: stale -->\n"
        "<!-- note_key: duplicate -->\nQ: Hello\n```\n"
        "<!-- note_type: example -->\n```\nA: Answer"
    )
    assert _upsert_note_metadata_in_block(block, note, "key") == (
        "<!-- note_key: key -->\n<!-- note_type: QA -->\nQ: Hello\n```\n"
        "<!-- note_type: example -->\n```\nA: Answer"
    )

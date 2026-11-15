"""Matching and ordering notes during export."""

from typing import Any

from ankiops.notes import Note
from ankiops.sync.from_anki import _order_resolved_notes


def _note(key: str | None, text: str) -> Note:
    return Note(note_key=key, note_type="QA", fields={"Question": text})


def test_keyed_notes_use_identity_and_new_notes_are_appended(monkeypatch: Any) -> None:
    existing = [_note("b", "old b"), _note("a", "old a")]
    a, b, c = _note("a", "new a"), _note("b", "new b"), _note("c", "new c")

    def unexpected_fingerprint(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Keyed notes should not require content matching")

    monkeypatch.setattr(
        "ankiops.sync.from_anki.note_fingerprint", unexpected_fingerprint
    )
    ordered, protected = _order_resolved_notes(
        [("a", 1, a), ("b", 2, b), ("c", 3, c)], existing, False
    )
    assert ordered == [b, a, c]
    assert protected == 0


def test_keyless_notes_match_content_or_are_preserved() -> None:
    a, b, c = _note("a", "A"), _note("b", "B"), _note("c", "C")
    draft = _note(None, "draft")
    existing = [_note("a", "old A"), _note(None, "B"), draft]

    ordered, protected = _order_resolved_notes(
        [("a", 1, a), ("b", 2, b), ("c", 3, c)], existing, False
    )
    assert ordered == [a, b, draft, c]
    assert protected == 1

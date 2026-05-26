"""Tests for note-level tag helpers."""

from ankiops.tags import format_tags_comment, normalize_tags, parse_tags_comment


def test_normalize_tags_handles_empty_input():
    assert normalize_tags(None) == ()
    assert normalize_tags("") == ()
    assert normalize_tags(["", "   "]) == ()


def test_normalize_tags_dedupes_trims_and_sorts():
    assert normalize_tags([" z ", "a", "z", "topic::subtopic"]) == (
        "a",
        "topic::subtopic",
        "z",
    )


def test_normalize_tags_accepts_whitespace_separated_string():
    assert normalize_tags("z a topic::subtopic a") == (
        "a",
        "topic::subtopic",
        "z",
    )


def test_parse_tags_comment():
    assert parse_tags_comment("<!-- tags: z a topic::subtopic -->") == (
        "a",
        "topic::subtopic",
        "z",
    )
    assert parse_tags_comment("<!-- tags: -->") == ()
    assert parse_tags_comment("Q: Not a tag comment") is None


def test_format_tags_comment_omits_empty_tags():
    assert format_tags_comment(()) is None
    assert format_tags_comment(["z", "a"]) == "<!-- tags: a z -->"

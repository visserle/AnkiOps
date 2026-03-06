"""Unit tests for note-type CLI helper validation functions."""

from __future__ import annotations

import pytest

from ankiops.note_type_cli import _parse_identifying_answer, _validate_prefix_input


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("y", True),
        ("Y", True),
        ("yes", True),
        ("Yes", True),
        ("n", False),
        ("N", False),
        ("no", False),
        ("No", False),
    ],
)
def test_parse_identifying_answer(value, expected):
    assert _parse_identifying_answer(value) is expected


def test_parse_identifying_answer_rejects_invalid():
    with pytest.raises(ValueError, match="Please enter 'y' or 'n'"):
        _parse_identifying_answer("maybe")


def test_validate_prefix_input_accepts_valid_prefix():
    result = _validate_prefix_input("Q1:", used_prefixes=set())
    assert result == "Q1:"


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "  ",
        "NoColon",
        "1X:",
        "A-B:",
    ],
)
def test_validate_prefix_input_rejects_invalid(prefix):
    with pytest.raises(ValueError):
        _validate_prefix_input(prefix, used_prefixes=set())


def test_validate_prefix_input_rejects_duplicates():
    with pytest.raises(ValueError, match="already used"):
        _validate_prefix_input("Q:", used_prefixes={"Q:"})

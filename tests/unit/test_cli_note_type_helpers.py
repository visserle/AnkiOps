"""Unit tests for note-type CLI helper validation functions."""

from __future__ import annotations

import pytest

from ankiops.note_type_cli import (
    _parse_identifying_answer,
    _validate_global_label_reuse,
    _validate_label_input,
)


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


def test_validate_label_input_accepts_valid_label():
    result = _validate_label_input("Q1", used_labels=set())
    assert result == "Q1:"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Q1:", "Q1:"),
        ("Q_1", "Q_1:"),
        ("Q-1", "Q-1:"),
    ],
)
def test_validate_label_input_accepts_optional_colon_and_extra_chars(label, expected):
    result = _validate_label_input(label, used_labels=set())
    assert result == expected


@pytest.mark.parametrize(
    "label",
    [
        "",
        "  ",
        ":",
        "A:B",
        "A B",
        "_X",
        "-X",
    ],
)
def test_validate_label_input_rejects_invalid(label):
    with pytest.raises(ValueError):
        _validate_label_input(label, used_labels=set())


def test_validate_label_input_rejects_duplicates():
    with pytest.raises(ValueError, match="already used"):
        _validate_label_input("Q", used_labels={"Q:"})


def test_validate_global_label_reuse_allows_consistent_reuse():
    _validate_global_label_reuse(
        label="Q:",
        field_name="Question",
        identifying=True,
        label_to_field_name={"Q:": "Question"},
        label_to_identifying={"Q:": True},
    )


def test_validate_global_label_reuse_rejects_field_name_mismatch():
    with pytest.raises(ValueError, match="already mapped to field"):
        _validate_global_label_reuse(
            label="Q:",
            field_name="Prompt",
            identifying=True,
            label_to_field_name={"Q:": "Question"},
            label_to_identifying={"Q:": True},
        )


def test_validate_global_label_reuse_rejects_identifying_mismatch():
    with pytest.raises(ValueError, match="already has IDENTIFYING=True"):
        _validate_global_label_reuse(
            label="Q:",
            field_name="Question",
            identifying=False,
            label_to_field_name={"Q:": "Question"},
            label_to_identifying={"Q:": True},
        )

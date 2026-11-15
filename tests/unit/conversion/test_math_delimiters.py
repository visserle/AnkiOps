"""Tests for shared math delimiter handling."""

import re

import pytest

from ankiops.math_delimiters import MATH_SEGMENT_RE, normalize_escaped_math_delimiters


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        (r"before \(x\) and \[y\] after", [r"\(x\)", r"\[y\]"]),
        ("\\[x\n+y\\]", ["\\[x\n+y\\]"]),
        (r"\\(x\\) \\\[y\]", [r"\\(x\\)", r"\\\[y\]"]),
        (r"\[\]\] \(\) \[unfinished", []),
        (r"\[x\]\] \(y\)\)", [r"\[x\]", r"\(y\)"]),
    ],
)
def test_math_segment_substitution(text: str, segments: list[str]) -> None:
    found: list[str] = []

    def replace(match: re.Match[str]) -> str:
        found.append(match.group(0))
        return f"<math{len(found)}>"

    result = MATH_SEGMENT_RE.sub(replace, text)

    assert found == segments
    expected = text
    for i, segment in enumerate(segments, 1):
        expected = expected.replace(segment, f"<math{i}>", 1)
    assert result == expected


def test_normalization_only_changes_segments_with_math_content_hints() -> None:
    text = r"\\(x\\) \\[y\\] \\(x_1\\) \\[\alpha\\]"

    assert normalize_escaped_math_delimiters(text) == (
        r"\\(x\\) \\[y\\] \(x_1\) \[\alpha\]"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"\\[\\(x_1\\)\\]", r"\[\(x_1\)\]"),
        (r"\\(\\[x_1\\]\\)", r"\(\[x_1\]\)"),
    ],
)
def test_normalization_processes_nested_delimiter_types(
    text: str, expected: str
) -> None:
    assert normalize_escaped_math_delimiters(text) == expected

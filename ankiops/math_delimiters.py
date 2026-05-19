"""Helpers for normalizing escaped LaTeX math delimiters."""

from __future__ import annotations

import re

_DOUBLE_ESCAPED_INLINE_MATH_PAREN_PATTERN = re.compile(
    r"\\{2,}\((?P<dipm_text>(?:(?!\\{2,}\))[\s\S])+?)\\{2,}\)"
)
_DOUBLE_ESCAPED_INLINE_MATH_BRACKET_PATTERN = re.compile(
    r"\\{2,}\[(?P<dibm_text>(?:(?!\\{2,}\])[\s\S])+?)\\{2,}\]"
)
_MATH_BODY_HINT_PATTERN = re.compile(r"(\\[A-Za-z]+|[_^{}])")


def normalize_double_escaped_math_delimiters(text: str) -> str:
    """Canonicalize doubly escaped math delimiters to single-escaped form."""

    def _replace_paren(match: re.Match[str]) -> str:
        body = match.group("dipm_text")
        if not _MATH_BODY_HINT_PATTERN.search(body):
            return match.group(0)
        return "\\(" + body + "\\)"

    def _replace_bracket(match: re.Match[str]) -> str:
        body = match.group("dibm_text")
        if not _MATH_BODY_HINT_PATTERN.search(body):
            return match.group(0)
        return "\\[" + body + "\\]"

    text = _DOUBLE_ESCAPED_INLINE_MATH_PAREN_PATTERN.sub(_replace_paren, text)
    text = _DOUBLE_ESCAPED_INLINE_MATH_BRACKET_PATTERN.sub(_replace_bracket, text)
    return text


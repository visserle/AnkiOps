"""Helpers for normalizing escaped LaTeX math delimiters."""

from __future__ import annotations

import re

_ESCAPED_INLINE_MATH_PAREN_PATTERN = re.compile(
    r"\\{1,}\((?P<ipm_text>(?:(?!\\{1,}\))[\s\S])+?)\\{1,}\)"
)
_ESCAPED_INLINE_MATH_BRACKET_PATTERN = re.compile(
    r"\\{1,}\[(?P<ibm_text>(?:(?!\\{1,}\])[\s\S])+?)\\{1,}\]"
)
_MATH_BODY_HINT_PATTERN = re.compile(r"(\\[A-Za-z]+|[_^{}])")


def normalize_escaped_math_delimiters(text: str) -> str:
    """Canonicalize escaped math delimiter runs to single-escaped form."""

    def _replace_paren(match: re.Match[str]) -> str:
        body = match.group("ipm_text")
        if not _MATH_BODY_HINT_PATTERN.search(body):
            return match.group(0)
        return "\\(" + body + "\\)"

    def _replace_bracket(match: re.Match[str]) -> str:
        body = match.group("ibm_text")
        if not _MATH_BODY_HINT_PATTERN.search(body):
            return match.group(0)
        return "\\[" + body + "\\]"

    text = _ESCAPED_INLINE_MATH_PAREN_PATTERN.sub(_replace_paren, text)
    text = _ESCAPED_INLINE_MATH_BRACKET_PATTERN.sub(_replace_bracket, text)
    return text

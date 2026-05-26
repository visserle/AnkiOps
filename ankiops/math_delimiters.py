"""Helpers for normalizing escaped LaTeX math delimiters."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_PAREN_DELIMITED_MATH_PATTERN = r"\\\((?P<paren_math_body>[\s\S]+?)\\\)"
_BRACKET_DELIMITED_MATH_PATTERN = r"\\\[(?P<bracket_math_body>[\s\S]+?)\\\]"
_BLOCK_BRACKET_DELIMITED_MATH_PATTERN = r"^\\\[(?P<block_math_body>[\s\S]+?)\\\][ \t]*$"
_ESCAPED_PAREN_DELIMITED_MATH_RE = re.compile(
    r"\\{1,}\((?P<escaped_paren_math_body>(?:(?!\\{1,}\))[\s\S])+?)\\{1,}\)"
)
_ESCAPED_BRACKET_DELIMITED_MATH_RE = re.compile(
    r"\\{1,}\[(?P<escaped_bracket_math_body>(?:(?!\\{1,}\])[\s\S])+?)\\{1,}\]"
)
_MATH_CONTENT_HINT_RE = re.compile(r"(\\[A-Za-z]+|[_^{}])")


def preserve_math_delimiters_plugin(md: Any) -> None:
    """Preserve LaTeX math delimiters \\(...\\) and \\[...\\] through Mistune."""

    def _parse_token(
        token_type: str, body_group_name: str
    ) -> Callable[[Any, re.Match[str], Any], int]:
        def parser(_: Any, regex_match: re.Match[str], state: Any) -> int:
            state.append_token(
                {"type": token_type, "raw": regex_match.group(body_group_name)}
            )
            return regex_match.end()

        return parser

    def _parse_block(_: Any, regex_match: re.Match[str], state: Any) -> int:
        state.append_token(
            {"type": "block_math", "raw": regex_match.group("block_math_body")}
        )
        return regex_match.end() + 1

    md.inline.register(
        "inline_math_paren",
        _PAREN_DELIMITED_MATH_PATTERN,
        _parse_token("inline_math_paren", "paren_math_body"),
        before="escape",
    )
    md.inline.register(
        "inline_math_bracket",
        _BRACKET_DELIMITED_MATH_PATTERN,
        _parse_token("inline_math_bracket", "bracket_math_body"),
        before="escape",
    )
    md.block.register(
        "block_math", _BLOCK_BRACKET_DELIMITED_MATH_PATTERN, _parse_block, before="list"
    )
    if md.renderer and md.renderer.NAME == "html":
        md.renderer.register(
            "inline_math_paren",
            lambda _, token_text: "\\(" + token_text + "\\)",
        )
        md.renderer.register(
            "inline_math_bracket",
            lambda _, token_text: "\\[" + token_text + "\\]",
        )
        md.renderer.register(
            "block_math", lambda _, token_text: "\\[" + token_text + "\\]"
        )


def normalize_escaped_math_delimiters(text: str) -> str:
    """Canonicalize escaped math delimiter runs to single-escaped form."""

    def _replace_paren(match: re.Match[str]) -> str:
        body = match.group("escaped_paren_math_body")
        if not _MATH_CONTENT_HINT_RE.search(body):
            return match.group(0)
        return "\\(" + body + "\\)"

    def _replace_bracket(match: re.Match[str]) -> str:
        body = match.group("escaped_bracket_math_body")
        if not _MATH_CONTENT_HINT_RE.search(body):
            return match.group(0)
        return "\\[" + body + "\\]"

    text = _ESCAPED_PAREN_DELIMITED_MATH_RE.sub(_replace_paren, text)
    text = _ESCAPED_BRACKET_DELIMITED_MATH_RE.sub(_replace_bracket, text)
    return text

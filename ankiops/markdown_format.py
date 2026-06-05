"""Shared Markdown format conventions for AnkiOps notes."""

from __future__ import annotations

import re

NOTE_SEPARATOR = "\n\n---\n\n"  # changing the whitespace might lead to issues

NOTE_KEY_COMMENT_RE = re.compile(
    r"^\s*<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->\s*$"
)
NOTE_TYPE_COMMENT_RE = re.compile(r"^\s*<!--\s*note_type:\s*(.*?)\s*-->\s*$")
TAGS_COMMENT_RE = re.compile(r"^\s*<!--\s*tags:\s*(.*?)\s*-->\s*$")
CODE_FENCE_RE = re.compile(r"^(```|~~~)")
FIELD_LABEL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
FIELD_LABEL_CANDIDATE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*:)(?:\s|$)")


def format_note_key_comment(note_key: str) -> str:
    return f"<!-- note_key: {note_key} -->"


def format_note_type_comment(note_type: str) -> str:
    return f"<!-- note_type: {note_type} -->"


def parse_note_key_comment(line: str) -> str | None:
    match = NOTE_KEY_COMMENT_RE.match(line)
    return match.group(1) if match else None


def parse_note_type_comment(line: str) -> str | None:
    match = NOTE_TYPE_COMMENT_RE.match(line)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def is_note_type_comment(line: str) -> bool:
    return NOTE_TYPE_COMMENT_RE.match(line) is not None


def is_code_fence_line(line: str) -> bool:
    return CODE_FENCE_RE.match(line) is not None

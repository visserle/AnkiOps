"""Validation utilities for AI JSON responses."""

from __future__ import annotations

import json
from typing import Any

from .errors import AIResponseError
from .types import InlineEditedNote


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text output."""
    body = content.strip()
    if not body:
        raise AIResponseError("AI response was empty")

    parsed = _try_parse_json_dict(body)
    if parsed is not None:
        return parsed

    fenced = _extract_json_fence(body)
    if fenced is not None:
        parsed = _try_parse_json_dict(fenced)
        if parsed is not None:
            return parsed

    candidate = _extract_first_json_object(body)
    if candidate is not None:
        parsed = _try_parse_json_dict(candidate)
        if parsed is not None:
            return parsed

    raise AIResponseError("AI response was not a valid JSON object")


def normalize_batch_response(content: str) -> dict[str, InlineEditedNote]:
    """Normalize supported batch response shapes into {note_key: note_json}."""
    parsed = parse_json_object(content)
    notes_payload = parsed.get("notes", parsed)

    if isinstance(notes_payload, list):
        normalized: dict[str, InlineEditedNote] = {}
        for raw_note in notes_payload:
            note = _normalize_note_object(raw_note)
            note_key = note.note_key
            if note_key in normalized:
                raise AIResponseError(
                    f"Batch response contains duplicate note_key '{note_key}'"
                )
            normalized[note_key] = note
        return normalized

    if not isinstance(notes_payload, dict):
        raise AIResponseError(
            "Batch response must contain a mapping or list in 'notes'"
        )

    if _looks_like_single_note(notes_payload):
        note = _normalize_note_object(notes_payload)
        return {note.note_key: note}

    normalized_by_key: dict[str, InlineEditedNote] = {}
    for raw_key, raw_note in notes_payload.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise AIResponseError(
                "Batch response keys must be non-empty note_key strings"
            )
        note = _normalize_note_object(raw_note, fallback_note_key=raw_key.strip())
        note_key = note.note_key
        if note_key in normalized_by_key:
            raise AIResponseError(
                f"Batch response contains duplicate note_key '{note_key}'"
            )
        normalized_by_key[note_key] = note
    return normalized_by_key


def _normalize_note_object(
    raw_note: Any,
    *,
    fallback_note_key: str | None = None,
) -> InlineEditedNote:
    if not isinstance(raw_note, dict):
        raise AIResponseError("Batch response notes must be JSON objects")

    raw_note_key = raw_note.get("note_key", fallback_note_key)
    if not isinstance(raw_note_key, str) or not raw_note_key.strip():
        raise AIResponseError("Batch response note is missing a valid note_key")
    note_key = raw_note_key.strip()

    fields = raw_note.get("fields")
    if not isinstance(fields, dict):
        raise AIResponseError(
            f"Batch response note '{note_key}' is missing a fields object"
        )

    string_fields: dict[str, str] = {}
    for field_name, field_value in fields.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise AIResponseError(
                f"Batch response note '{note_key}' has invalid field name"
            )
        if not isinstance(field_value, str):
            raise AIResponseError(
                f"Batch response note '{note_key}' has non-string field '{field_name}'"
            )
        string_fields[field_name] = field_value

    raw_note_type = raw_note.get("note_type")
    note_type = None
    if raw_note_type is not None:
        if not isinstance(raw_note_type, str) or not raw_note_type.strip():
            raise AIResponseError(
                f"Batch response note '{note_key}' has invalid note_type"
            )
        note_type = raw_note_type.strip()

    return InlineEditedNote.from_parts(
        note_key=note_key,
        fields=string_fields,
        note_type=note_type,
    )


def _looks_like_single_note(payload: dict[str, Any]) -> bool:
    return "note_key" in payload and "fields" in payload


def _try_parse_json_dict(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_json_fence(content: str) -> str | None:
    prefix = "```json"
    lower = content.lower()
    start = lower.find(prefix)
    if start == -1:
        start = content.find("```")
        if start == -1:
            return None
        fence_start = start + 3
    else:
        fence_start = start + len(prefix)

    end = content.find("```", fence_start)
    if end == -1:
        return None
    return content[fence_start:end].strip()


def _extract_first_json_object(content: str) -> str | None:
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(content[start:], start):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = False
                continue
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    return None

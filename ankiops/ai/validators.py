"""Validation utilities for AI JSON responses."""

from __future__ import annotations

import json
from typing import Any


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text output."""
    body = content.strip()
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first_brace = body.find("{")
    last_brace = body.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidate = body[first_brace : last_brace + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("AI response was not a valid JSON object")


def normalize_batch_response(content: str) -> dict[str, dict[str, Any]]:
    """Normalize several supported batch JSON response shapes by note_key."""
    parsed = parse_json_object(content)
    notes_payload = parsed.get("notes", parsed)

    if isinstance(notes_payload, list):
        normalized: dict[str, dict[str, Any]] = {}
        for item in notes_payload:
            if not isinstance(item, dict):
                raise ValueError("Batch response list must contain JSON objects")
            note_key = item.get("note_key")
            if not isinstance(note_key, str) or not note_key.strip():
                raise ValueError("Batch response note is missing a valid note_key")
            if note_key in normalized:
                raise ValueError(
                    f"Batch response contains duplicate note_key '{note_key}'"
                )
            normalized[note_key] = item
        return normalized

    if not isinstance(notes_payload, dict):
        raise ValueError("Batch response must contain a mapping or list in 'notes'")

    if "note_key" in notes_payload and "fields" in notes_payload:
        note_key = notes_payload.get("note_key")
        if not isinstance(note_key, str) or not note_key.strip():
            raise ValueError("Batch response note is missing a valid note_key")
        return {note_key: notes_payload}

    normalized: dict[str, dict[str, Any]] = {}
    for response_key, response_value in notes_payload.items():
        if not isinstance(response_key, str) or not response_key.strip():
            raise ValueError("Batch response keys must be non-empty note_key strings")
        if not isinstance(response_value, dict):
            raise ValueError("Batch response map values must be JSON objects")

        model_note_key = response_value.get("note_key")
        note_key = (
            model_note_key
            if isinstance(model_note_key, str) and model_note_key.strip()
            else response_key
        )
        if note_key in normalized:
            raise ValueError(f"Batch response contains duplicate note_key '{note_key}'")
        normalized[note_key] = response_value
    return normalized

"""Dynamic Pydantic response models for OpenAI structured outputs."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import ConfigDict, create_model

from .types import NotePayload

_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def build_response_model(
    *,
    note_type: str,
    payloads: list[NotePayload],
) -> type[Any]:
    """Build a strict response model for one note type and request batch."""
    if not payloads:
        raise ValueError("response model requires at least one payload")

    note_keys = sorted({payload.note_key for payload in payloads})
    editable_fields = sorted(
        {field_name for payload in payloads for field_name in payload.editable_fields}
    )
    if not editable_fields:
        raise ValueError("response model requires at least one editable field")

    suffix = _safe_type_name(note_type)
    update_model = create_model(
        f"{suffix}Update",
        __config__=ConfigDict(extra="forbid"),
        note_key=(_literal(note_keys), ...),
        field=(_literal(editable_fields), ...),
        value=(str, ...),
    )
    return create_model(
        f"{suffix}Response",
        __config__=ConfigDict(extra="forbid"),
        updates=(list[update_model], ...),
    )


def parsed_updates(parsed_response: object) -> list[tuple[str, str, str]]:
    """Return ``(note_key, field, value)`` tuples from a parsed Pydantic object."""
    raw_updates = getattr(parsed_response, "updates", None)
    if not isinstance(raw_updates, list):
        raise ValueError("Parsed response is missing updates list")

    updates: list[tuple[str, str, str]] = []
    for raw_update in raw_updates:
        note_key = getattr(raw_update, "note_key", None)
        field = getattr(raw_update, "field", None)
        value = getattr(raw_update, "value", None)
        if not isinstance(note_key, str):
            raise ValueError("Parsed update note_key must be a string")
        if not isinstance(field, str):
            raise ValueError("Parsed update field must be a string")
        if not isinstance(value, str):
            raise ValueError("Parsed update value must be a string")
        updates.append((note_key, field, value))
    return updates


def parsed_response_json(parsed_response: object) -> dict[str, object] | None:
    model_dump = getattr(parsed_response, "model_dump", None)
    if not callable(model_dump):
        return None
    value = model_dump(mode="json")
    return value if isinstance(value, dict) else None


def _literal(values: list[str]) -> object:
    if not values:
        raise ValueError("Literal requires at least one value")
    return Literal.__getitem__(tuple(values))


def _safe_type_name(value: str) -> str:
    safe = _SAFE_NAME_PATTERN.sub("_", value).strip("_")
    if not safe:
        return "Note"
    if safe[0].isdigit():
        safe = f"Note_{safe}"
    return safe

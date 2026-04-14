"""Shared structured-output helpers for note update responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .task_types import NotePayload, NoteUpdate


@dataclass(frozen=True)
class NoteUpdateContract:
    """Shared contract for structured note update responses."""

    schema: dict[str, object]
    editable_fields: frozenset[str]


class StructuredOutputError(ValueError):
    """Raised when a model response violates the note update contract."""


def build_note_update_contract(note_payload: NotePayload) -> NoteUpdateContract:
    """Build the shared JSON schema contract for a note update response."""
    editable_fields = tuple(note_payload.editable_fields.keys())
    edit_properties = {field_name: {"type": "string"} for field_name in editable_fields}
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "note_key": {"type": "string"},
            "edits": {
                "type": "object",
                "properties": edit_properties,
                "additionalProperties": False,
            },
        },
        "required": ["note_key", "edits"],
        "additionalProperties": False,
    }
    return NoteUpdateContract(
        schema=schema,
        editable_fields=frozenset(editable_fields),
    )


def parse_note_update_json(
    raw_text: str,
    *,
    contract: NoteUpdateContract,
) -> NoteUpdate:
    """Parse and validate a JSON note update response."""
    try:
        data = json.loads(raw_text)
    except ValueError as error:
        raise StructuredOutputError("Response was not valid JSON") from error
    return validate_note_update_data(data, contract=contract)


def validate_note_update_data(
    data: object,
    *,
    contract: NoteUpdateContract,
) -> NoteUpdate:
    """Validate a parsed note update response against the shared contract."""
    if not isinstance(data, Mapping):
        _raise_validation_error("response must be an object")

    _reject_unknown_top_level_keys(data)

    note_key = data.get("note_key")
    if not isinstance(note_key, str):
        _raise_validation_error("note_key must be a string")

    edits = data.get("edits")
    if not isinstance(edits, Mapping):
        _raise_validation_error("edits must be an object")

    parsed_edits: dict[str, str] = {}
    for field_name, value in edits.items():
        if not isinstance(field_name, str):
            _raise_validation_error("edits keys must be strings")
        if field_name not in contract.editable_fields:
            _raise_validation_error(f"edits.{field_name} is not editable")
        if not isinstance(value, str):
            _raise_validation_error(f"edits.{field_name} must be a string")
        parsed_edits[field_name] = value

    return NoteUpdate(note_key=note_key, edits=parsed_edits)


def _reject_unknown_top_level_keys(data: Mapping[object, object]) -> None:
    allowed_keys = {"note_key", "edits"}
    for key in data:
        if not isinstance(key, str):
            _raise_validation_error("top-level keys must be strings")
        if key not in allowed_keys:
            _raise_validation_error(f"{key} is not allowed")


def _raise_validation_error(message: str) -> None:
    raise StructuredOutputError(f"Structured output validation failed: {message}")

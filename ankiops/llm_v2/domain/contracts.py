"""Structured output contract definitions for runtime v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import ContractValidationError
from .payloads import NotePayload, NoteUpdate

_ALLOWED_TOP_LEVEL_KEYS = {"note_key", "edits"}


@dataclass(frozen=True)
class NoteUpdateContract:
    schema_name: str
    json_schema: dict[str, object]
    editable_fields: frozenset[str]
    fingerprint: str

    def parse_raw_json(self, raw_text: str) -> NoteUpdate:
        try:
            data = json.loads(raw_text)
        except ValueError as error:
            raise ContractValidationError("Response was not valid JSON") from error
        return self.parse_data(data)

    def parse_data(self, data: object) -> NoteUpdate:
        if not isinstance(data, Mapping):
            raise ContractValidationError("response must be an object")

        self._reject_unknown_top_level_keys(data)

        note_key = data.get("note_key")
        if not isinstance(note_key, str):
            raise ContractValidationError("note_key must be a string")

        edits = data.get("edits")
        if not isinstance(edits, Mapping):
            raise ContractValidationError("edits must be an object")

        parsed_edits: dict[str, str] = {}
        for field_name, value in edits.items():
            if not isinstance(field_name, str):
                raise ContractValidationError("edits keys must be strings")
            if field_name not in self.editable_fields:
                raise ContractValidationError(f"edits.{field_name} is not editable")
            if not isinstance(value, str):
                raise ContractValidationError(
                    f"edits.{field_name} must be a string"
                )
            parsed_edits[field_name] = value

        return NoteUpdate(note_key=note_key, edits=parsed_edits)

    @staticmethod
    def _reject_unknown_top_level_keys(data: Mapping[object, object]) -> None:
        for key in data:
            if not isinstance(key, str):
                raise ContractValidationError("top-level keys must be strings")
            if key not in _ALLOWED_TOP_LEVEL_KEYS:
                raise ContractValidationError(f"{key} is not allowed")


def build_note_update_contract(note_payload: NotePayload) -> NoteUpdateContract:
    editable_fields = tuple(note_payload.editable_fields.keys())
    edit_properties = {field_name: {"type": "string"} for field_name in editable_fields}
    json_schema: dict[str, object] = {
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
    fingerprint = _fingerprint_schema(json_schema)
    return NoteUpdateContract(
        schema_name="note_update",
        json_schema=json_schema,
        editable_fields=frozenset(editable_fields),
        fingerprint=fingerprint,
    )


def _fingerprint_schema(schema: dict[str, object]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8"))
    return digest.hexdigest()

"""Validation utilities for AI JSON responses."""

import json
from typing import Any

from ankiops.ai.errors import AIResponseError


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text output."""
    body = content.strip()
    if not body:
        raise AIResponseError("AI response was empty")

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        raise AIResponseError(
            "AI response must be a strict JSON object with no surrounding text"
        ) from error

    if not isinstance(parsed, dict):
        raise AIResponseError("AI response body must be a JSON object")
    return parsed


def normalize_patch_response(content: str) -> dict[str, dict[str, str]]:
    """Normalize AI patch response into {ref: {field_name: edited_text}}."""
    parsed = parse_json_object(content)
    patches = parsed.get("patches")
    if not isinstance(patches, list):
        raise AIResponseError("Patch response must contain a top-level 'patches' list")
    if not patches:
        raise AIResponseError("Patch response patches list cannot be empty")

    normalized: dict[str, dict[str, str]] = {}
    for raw_patch in patches:
        if not isinstance(raw_patch, dict):
            raise AIResponseError("Patch entries must be JSON objects")

        raw_ref = raw_patch.get("ref")
        if not isinstance(raw_ref, str) or not raw_ref.strip():
            raise AIResponseError("Patch entry is missing a valid ref")
        ref = raw_ref.strip()
        if ref in normalized:
            raise AIResponseError(f"Patch response contains duplicate ref '{ref}'")

        fields = raw_patch.get("fields")
        if not isinstance(fields, dict):
            raise AIResponseError(f"Patch entry '{ref}' is missing a fields object")

        string_fields: dict[str, str] = {}
        for field_name, field_value in fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise AIResponseError(f"Patch entry '{ref}' has invalid field name")
            if not isinstance(field_value, str):
                raise AIResponseError(
                    f"Patch entry '{ref}' has non-string field '{field_name}'"
                )
            string_fields[field_name] = field_value

        normalized[ref] = string_fields
    return normalized

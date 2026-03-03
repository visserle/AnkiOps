"""Prompt helpers for the LLM task runner."""

from __future__ import annotations

import json

from .models import NotePayload

_SYSTEM_PROMPT = """You are editing a single serialized Anki note.
Return JSON only.
The JSON must match the provided schema exactly.
Repeat the input note_key exactly.
Only modify fields listed in editable_fields.
Do not modify read_only_fields.
Do not invent new field names or change field names.
Preserve Markdown structure, math, code fences, links, cloze syntax, and meaning.
Return only changed editable fields in edits.
If no changes are needed, return an empty edits object."""


def build_instructions(task_prompt: str) -> str:
    """Build the final instructions sent to the model."""
    return f"{_SYSTEM_PROMPT}\n\nTask instructions:\n{task_prompt.strip()}"


def build_user_payload(note_payload: NotePayload) -> str:
    """Serialize the note payload for the model request."""
    payload: dict[str, object] = {
        "note_key": note_payload.note_key,
        "note_type": note_payload.note_type,
        "editable_fields": note_payload.editable_fields,
    }
    if note_payload.read_only_fields:
        payload["read_only_fields"] = note_payload.read_only_fields
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_note_patch_schema(note_payload: NotePayload) -> dict[str, object]:
    """Build a JSON Schema for a note patch response.

    The schema includes the exact editable field names so models produce
    tightly constrained output.
    """
    edit_field_names = list(note_payload.editable_fields.keys())
    edit_properties: dict[str, object] = {
        field_name: {"type": ["string", "null"]} for field_name in edit_field_names
    }
    return {
        "type": "object",
        "properties": {
            "note_key": {"type": "string"},
            "edits": {
                "type": "object",
                "properties": edit_properties,
                "required": edit_field_names,
                "additionalProperties": False,
            },
        },
        "required": ["note_key", "edits"],
        "additionalProperties": False,
    }

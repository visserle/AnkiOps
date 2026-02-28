"""Prompt helpers for the LLM task runner."""

from __future__ import annotations

import json

from .models import NotePayload

NOTE_PATCH_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "note_key": {"type": "string"},
        "edits": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["note_key", "edits"],
    "additionalProperties": False,
}

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

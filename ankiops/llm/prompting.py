"""Prompt helpers for Claude task requests."""

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
If no changes are needed, return an empty edits object.
Do not use null values anywhere in edits.
Use an empty string only when you intentionally want to clear a field."""


def build_system_prompt() -> str:
    """Build the stable Claude system prompt."""
    return _SYSTEM_PROMPT


def build_user_message(task_prompt: str, note_payload: NotePayload) -> str:
    """Build the user turn with explicit task and note context."""
    payload: dict[str, object] = {
        "note_key": note_payload.note_key,
        "note_type": note_payload.note_type,
        "editable_fields": note_payload.editable_fields,
    }
    if note_payload.read_only_fields:
        payload["read_only_fields"] = note_payload.read_only_fields
    note_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"<task>\n{task_prompt.strip()}\n</task>\n\n<note>\n{note_json}\n</note>"

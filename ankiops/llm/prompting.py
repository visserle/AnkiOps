"""Prompt helpers for Claude task requests."""

from __future__ import annotations

import json

from .llm_models import NotePayload


def build_system_prompt(system_prompt: str) -> str:
    """Build the Claude system prompt from task configuration."""
    return system_prompt.strip()


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

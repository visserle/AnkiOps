"""Task result validation and mutation helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ankiops.ai.types import InlineEditedNote, TaskChange, TaskRunResult

if TYPE_CHECKING:
    from ankiops.ai.task_selection import NoteTask

_FIELD_SAMPLE_LIMIT = 2
_FIELD_PREVIEW_MAX_CHARS = 40
_CLOZE_PATTERN = re.compile(r"\{\{c\d+::")


def validate_edited_note(
    note_task: NoteTask,
    edited_note: InlineEditedNote,
) -> str | None:
    """Validate one edited note against expected output shape."""
    if edited_note.note_key != note_task.note_key:
        return "response note_key mismatch"

    invalid_field_names = [
        field_name
        for field_name in edited_note.fields
        if field_name not in note_task.write_fields
    ]
    if invalid_field_names:
        invalid_fields = _format_field_samples(invalid_field_names)
        expected_fields = ", ".join(
            _field_preview(field_name) for field_name in note_task.write_fields
        )
        return (
            f"response returned {len(invalid_field_names)} unexpected field key(s): "
            f"{invalid_fields}; expected write_fields: {expected_fields}; "
            "hint: ensure patch keys exactly match write_fields"
        )

    non_string_fields = [
        field_name
        for field_name, value in edited_note.fields.items()
        if not isinstance(value, str)
    ]
    if non_string_fields:
        invalid_fields = _format_field_samples(non_string_fields)
        return (
            "response returned non-string value(s) for "
            f"{invalid_fields}; hint: all patched field values must be JSON strings"
        )

    if _is_cloze_note_type(note_task.note_type):
        if not _projected_fields_keep_cloze_marker(note_task, edited_note):
            return (
                "response removed or invalidated cloze syntax; "
                "cloze notes must contain at least one marker like "
                "'{{c1::answer}}'; hint: preserve cloze markers exactly"
            )
    return None


def _format_field_samples(field_names: list[str]) -> str:
    sampled = [
        _field_preview(field_name)
        for field_name in field_names[:_FIELD_SAMPLE_LIMIT]
    ]
    joined = ", ".join(sampled)
    remaining = len(field_names) - _FIELD_SAMPLE_LIMIT
    if remaining > 0:
        return f"{joined} (+{remaining} more)"
    return joined


def _field_preview(field_name: str) -> str:
    normalized = " ".join(field_name.split())
    if len(normalized) > _FIELD_PREVIEW_MAX_CHARS:
        normalized = normalized[: _FIELD_PREVIEW_MAX_CHARS - 3].rstrip() + "..."
    return f"'{normalized}'"


def _is_cloze_note_type(note_type: str) -> bool:
    return "cloze" in note_type.casefold()


def _projected_fields_keep_cloze_marker(
    note_task: NoteTask,
    edited_note: InlineEditedNote,
) -> bool:
    projected_fields = {
        field_name: field_value
        for field_name, field_value in note_task.note_fields.items()
        if isinstance(field_name, str) and isinstance(field_value, str)
    }
    projected_fields.update(edited_note.fields)
    return any(
        _CLOZE_PATTERN.search(field_value) for field_value in projected_fields.values()
    )


def apply_note_changes(
    note_task: NoteTask,
    edited_note: InlineEditedNote,
    task_id: str,
    result: TaskRunResult,
) -> bool:
    """Apply patch-style field edits to one note and collect change metadata."""
    changed_any = False
    for field_name, edited_text in edited_note.fields.items():
        original_text = note_task.original_write_fields[field_name]
        if edited_text == original_text:
            continue

        note_task.note_fields[field_name] = edited_text
        result.changed_fields += 1
        result.changes.append(
            TaskChange(
                task_id=task_id,
                deck_name=note_task.deck_name,
                note_key=note_task.note_key,
                note_type=note_task.note_type,
                field_name=field_name,
                original_text=original_text,
                edited_text=edited_text,
            )
        )
        changed_any = True
    return changed_any


def add_warning(
    result: TaskRunResult,
    warning: str,
    *,
    max_warnings: int,
) -> None:
    """Append warning while enforcing max warning cap."""
    if len(result.warnings) < max_warnings:
        result.warnings.append(warning)
        return
    result.dropped_warnings += 1

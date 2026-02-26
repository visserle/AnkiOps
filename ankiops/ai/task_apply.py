"""Task result validation and mutation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import InlineEditedNote, TaskChange, TaskRunResult

if TYPE_CHECKING:
    from .task_selection import NoteTask


def validate_edited_note(
    note_task: NoteTask,
    edited_note: InlineEditedNote,
) -> str | None:
    """Validate one edited note against requested task constraints."""
    if edited_note.note_key != note_task.note_key:
        return "response note_key mismatch"

    invalid_fields = [
        field_name
        for field_name in note_task.write_fields
        if not isinstance(edited_note.fields.get(field_name), str)
    ]
    if invalid_fields:
        return f"response fields invalid: {', '.join(invalid_fields)}"
    return None


def apply_note_changes(
    note_task: NoteTask,
    edited_note: InlineEditedNote,
    task_id: str,
    result: TaskRunResult,
) -> bool:
    """Apply changed write-fields to one note and collect change metadata."""
    changed_any = False
    for field_name in note_task.write_fields:
        original_text = note_task.original_write_fields[field_name]
        edited_text = edited_note.fields[field_name]
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

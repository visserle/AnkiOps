"""Shared command error messages."""

from __future__ import annotations


def format_missing_note_keys_error(missing_count: int) -> str:
    note_label = "note" if missing_count == 1 else "notes"
    return (
        f"Missing note_keys for {missing_count} {note_label}. "
        "note_keys are stable IDs AnkiOps needs to match notes across "
        "collections without duplicates. "
        "Fix: run 'ankiops fa' to assign them."
    )

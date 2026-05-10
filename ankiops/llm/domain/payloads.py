"""Typed payload objects for runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NotePayload:
    note_key: str
    note_type: str
    editable_fields: dict[str, str]
    read_only_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NoteUpdate:
    note_key: str
    edits: dict[str, str]

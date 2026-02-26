"""Shared data types for AI task execution."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ModelProfile:
    """A named model runtime profile loaded from models YAML."""

    name: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_in_flight: int


@dataclass(frozen=True)
class ModelsConfig:
    """Collection-scoped set of named model profiles."""

    default_profile: str
    profiles: dict[str, ModelProfile]
    source_path: Path | None = None


@dataclass(frozen=True)
class RuntimeAIConfig:
    """Resolved runtime model settings after applying profile + CLI overrides."""

    profile: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_in_flight: int
    api_key: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    """Task definition loaded from task YAML."""

    id: str
    instructions: str
    read_fields: list[str]
    write_fields: list[str]
    scope_note_types: list[str]
    scope_decks: list[str]
    scope_subdecks: bool
    batch: str
    batch_size: int
    model: str | None = None
    description: str = ""
    temperature: float = 0.0
    source_path: Path | None = None

    def matches_note_type(self, note_type: str) -> bool:
        """Whether this task applies to the given note type name."""
        return any(
            fnmatch.fnmatchcase(note_type, pattern) for pattern in self.scope_note_types
        )


@dataclass(frozen=True)
class TaskRunOptions:
    """Execution controls for task runs."""

    include_decks: list[str] | None = None
    batch_size: int | None = None
    max_in_flight: int = 4
    max_warnings: int = 200


@dataclass(frozen=True)
class TaskChange:
    """A single field mutation produced by a task run."""

    task_id: str
    deck_name: str
    note_key: str
    note_type: str
    field_name: str
    original_text: str
    edited_text: str


@dataclass
class TaskRunResult:
    """Aggregate task run outcome and diagnostics."""

    processed_decks: int = 0
    processed_notes: int = 0
    matched_notes: int = 0
    changed_fields: int = 0
    changes: list[TaskChange] = field(default_factory=list)
    changed_decks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_warnings: int = 0


@dataclass(frozen=True)
class InlineNotePayload:
    """Canonical payload sent to the AI editor for one note."""

    note_key: str
    note_type: str
    fields: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible object."""
        return {
            "note_key": self.note_key,
            "note_type": self.note_type,
            "fields": dict(self.fields),
        }


@dataclass(frozen=True)
class InlineEditedNote:
    """Canonical edited-note payload returned by the AI editor."""

    note_key: str
    fields: dict[str, str]
    note_type: str | None = None

    @classmethod
    def from_parts(
        cls,
        *,
        note_key: str,
        fields: Mapping[str, str],
        note_type: str | None = None,
    ) -> InlineEditedNote:
        """Build an edited note from validated parts."""
        normalized_note_key = note_key.strip()
        normalized_note_type = note_type.strip() if isinstance(note_type, str) else None
        return cls(
            note_key=normalized_note_key,
            fields=dict(fields),
            note_type=normalized_note_type or None,
        )


class AsyncInlineBatchEditor(Protocol):
    """Batch-capable async inline note editor."""

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        """Return edited notes keyed by note_key."""

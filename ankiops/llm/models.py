"""Typed models for the AnkiOps LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase


class FieldAccess(Enum):
    EDIT = "edit"
    READ_ONLY = "read_only"
    HIDDEN = "hidden"


class ProviderType(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass(frozen=True)
class DeckScope:
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=list)

    def matches(self, deck_name: str) -> bool:
        return any(fnmatchcase(deck_name, pattern) for pattern in self.include) and (
            not any(fnmatchcase(deck_name, pattern) for pattern in self.exclude)
        )


@dataclass(frozen=True)
class FieldExceptionRule:
    note_types: list[str] = field(default_factory=lambda: ["*"])
    read_only: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)

    def matches_note_type(self, note_type: str) -> bool:
        return any(fnmatchcase(note_type, pattern) for pattern in self.note_types)


@dataclass(frozen=True)
class TaskRequestOptions:
    temperature: float | None = None
    max_output_tokens: int | None = None

    def merged(self, override: "TaskRequestOptions") -> "TaskRequestOptions":
        return TaskRequestOptions(
            temperature=(
                override.temperature
                if override.temperature is not None
                else self.temperature
            ),
            max_output_tokens=(
                override.max_output_tokens
                if override.max_output_tokens is not None
                else self.max_output_tokens
            ),
        )


@dataclass(frozen=True)
class TaskConfig:
    version: int
    name: str
    provider: str
    prompt: str
    decks: DeckScope = field(default_factory=DeckScope)
    field_exceptions: list[FieldExceptionRule] = field(default_factory=list)
    request: TaskRequestOptions = field(default_factory=TaskRequestOptions)

    def field_access(self, note_type: str, field_name: str) -> FieldAccess:
        access = FieldAccess.EDIT
        for rule in self.field_exceptions:
            if not rule.matches_note_type(note_type):
                continue
            if field_name in rule.read_only and access == FieldAccess.EDIT:
                access = FieldAccess.READ_ONLY
            if field_name in rule.hidden:
                access = FieldAccess.HIDDEN
        return access


@dataclass(frozen=True)
class ProviderConfig:
    version: int
    name: str
    type: ProviderType
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: int = 60
    request_defaults: TaskRequestOptions = field(default_factory=TaskRequestOptions)


@dataclass(frozen=True)
class NotePayload:
    note_key: str
    deck_name: str
    note_type: str
    editable_fields: dict[str, str]
    read_only_fields: dict[str, str]


@dataclass(frozen=True)
class NotePatch:
    note_key: str
    updated_fields: dict[str, str]


@dataclass
class TaskRunSummary:
    task_name: str
    provider_name: str
    provider_type: ProviderType
    model: str
    eligible: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: int = 0

    def format(self) -> str:
        parts = [f"{self.updated} updated", f"{self.unchanged} unchanged"]
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if self.errors:
            parts.append(f"{self.errors} errors")
        return f"LLM task '{self.task_name}': {self.eligible} notes — " + ", ".join(
            parts
        )


@dataclass(frozen=True)
class LlmConfigSet:
    providers_by_name: dict[str, ProviderConfig]
    tasks_by_name: dict[str, TaskConfig]
    provider_errors: dict[str, str]
    task_errors: dict[str, str]

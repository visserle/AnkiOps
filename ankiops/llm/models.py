"""Typed models for the AnkiOps Claude task pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase

from ankiops.log import format_changes

from .anthropic_models import AnthropicModel


class FieldAccess(Enum):
    EDIT = "edit"
    READ_ONLY = "read_only"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class DeckScope:
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=list)
    include_subdecks: bool = True

    @staticmethod
    def _matches_pattern(
        deck_name: str,
        pattern: str,
        *,
        include_subdecks: bool,
    ) -> bool:
        if any(char in pattern for char in ("*", "?", "[")):
            return fnmatchcase(deck_name, pattern)

        if deck_name == pattern:
            return True
        if include_subdecks and deck_name.startswith(f"{pattern}::"):
            return True
        return False

    def matches(self, deck_name: str) -> bool:
        return any(
            self._matches_pattern(
                deck_name,
                pattern,
                include_subdecks=self.include_subdecks,
            )
            for pattern in self.include
        ) and (
            not any(
                self._matches_pattern(
                    deck_name,
                    pattern,
                    include_subdecks=self.include_subdecks,
                )
                for pattern in self.exclude
            )
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


@dataclass(frozen=True)
class TaskConfig:
    name: str
    model: AnthropicModel
    prompt: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_seconds: int = 60
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
class NotePayload:
    note_key: str
    note_type: str
    editable_fields: dict[str, str]
    read_only_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NoteUpdate:
    note_key: str
    edits: dict[str, str]


@dataclass(frozen=True)
class GenerateUpdateResult:
    update: NoteUpdate
    message_id: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass
class TaskRunSummary:
    task_name: str
    model: AnthropicModel
    decks_seen: int = 0
    decks_matched: int = 0
    notes_seen: int = 0
    eligible: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_deck_scope: int = 0
    skipped_no_editable_fields: int = 0
    errors: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider_latency_ms_total: int = 0

    @property
    def skipped(self) -> int:
        return self.skipped_deck_scope + self.skipped_no_editable_fields

    def format(self) -> str:
        return (
            f"LLM task '{self.task_name}' ({self.model}): {self.eligible} notes — "
            + format_changes(
                updated=self.updated,
                unchanged=self.unchanged,
                skipped=self.skipped,
                errors=self.errors,
            )
        )

    def format_usage(self) -> str:
        request_label = "request" if self.requests == 1 else "requests"
        provider_seconds = self.provider_latency_ms_total / 1000
        return (
            f"LLM usage: {self.requests} {request_label}, "
            f"{self.input_tokens} input tokens, "
            f"{self.output_tokens} output tokens, "
            f"{provider_seconds:.1f}s provider time"
        )

    def format_cost(self) -> str:
        estimate = self.model.estimate_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        return estimate.format()


@dataclass(frozen=True)
class TaskCatalog:
    tasks_by_name: dict[str, TaskConfig]
    errors: dict[str, str]

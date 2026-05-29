"""Small data types for the OpenAI-only LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ankiops.models import NoteTypeConfig, SyncSummary

from .model_registry import ModelSpec


class LlmItemStatus(Enum):
    QUEUED = "queued"
    SKIPPED_NO_EDITABLE_FIELDS = "skipped_no_editable_fields"
    INVALID_NOTE = "invalid_note"
    SUCCEEDED_UPDATED = "succeeded_updated"
    SUCCEEDED_UNCHANGED = "succeeded_unchanged"
    NOTE_ERROR = "note_error"
    PROVIDER_ERROR = "provider_error"
    FATAL_ERROR = "fatal_error"
    CANCELED = "canceled"


class LlmJobStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FieldAccess(Enum):
    EDITABLE = "editable"
    READ_ONLY = "read_only"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class DeckScope:
    deck_root: str | None = None

    def matches(self, deck_name: str) -> bool:
        if self.deck_root is None:
            return True
        return deck_name == self.deck_root or deck_name.startswith(
            f"{self.deck_root}::"
        )


@dataclass(frozen=True)
class FieldAccessRule:
    note_types: list[str] = field(default_factory=lambda: ["*"])
    editable: list[str] = field(default_factory=list)
    read_only: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)

    def matches_note_type(self, note_type: str) -> bool:
        return any(fnmatchcase(note_type, pattern) for pattern in self.note_types)

    def marks_editable(self, field_name: str) -> bool:
        return _matches_any(self.editable, field_name)

    def marks_read_only(self, field_name: str) -> bool:
        return _matches_any(self.read_only, field_name)

    def marks_hidden(self, field_name: str) -> bool:
        return _matches_any(self.hidden, field_name)


@dataclass(frozen=True)
class TaskRequestOptions:
    max_notes_per_request: int = 1
    temperature: float | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    name: str
    model: ModelSpec
    system_prompt: str
    user_prompt: str
    system_prompt_path: Path | None = None
    user_prompt_path: Path | None = None
    decks: DeckScope = field(default_factory=DeckScope)
    default_field_access: FieldAccess = FieldAccess.EDITABLE
    field_rules: list[FieldAccessRule] = field(default_factory=list)
    tag_access: FieldAccess = FieldAccess.HIDDEN
    request: TaskRequestOptions = field(default_factory=TaskRequestOptions)

    def field_access(self, note_type: str, field_name: str) -> FieldAccess:
        editable = False
        read_only = False
        hidden = False
        for rule in self.field_rules:
            if not rule.matches_note_type(note_type):
                continue
            editable = editable or rule.marks_editable(field_name)
            read_only = read_only or rule.marks_read_only(field_name)
            hidden = hidden or rule.marks_hidden(field_name)

        if hidden:
            return FieldAccess.HIDDEN
        if editable:
            return FieldAccess.EDITABLE
        if read_only:
            return FieldAccess.READ_ONLY
        return self.default_field_access


@dataclass(frozen=True)
class NotePayload:
    note_key: str
    note_type: str
    editable_fields: dict[str, str]
    read_only_fields: dict[str, str] = field(default_factory=dict)
    editable_tags: tuple[str, ...] | None = None
    read_only_tags: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DiscoveryCounts:
    decks_seen: int
    decks_matched: int
    notes_seen: int


@dataclass(frozen=True)
class DiscoveryItem:
    ordinal: int
    deck_name: str
    note_key: str | None
    note_type: str | None
    item_status: LlmItemStatus
    skip_reason: str | None
    error_message: str | None
    payload: NotePayload | None
    note_type_config: NoteTypeConfig | None
    serialized_note: dict[str, Any] | None


@dataclass(frozen=True)
class DiscoverySnapshot:
    counts: DiscoveryCounts
    items: list[DiscoveryItem]


@dataclass(frozen=True)
class EligibleCandidate:
    item_id: int
    deck_name: str
    payload: NotePayload
    note_type_config: NoteTypeConfig
    serialized_note: dict[str, Any]


@dataclass
class TaskRunSummary:
    task_name: str
    model: ModelSpec
    decks_seen: int = 0
    decks_matched: int = 0
    notes_seen: int = 0
    eligible: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_no_editable_fields: int = 0
    errors: int = 0
    canceled: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    provider_latency_ms_total: int = 0

    @property
    def skipped(self) -> int:
        return self.skipped_no_editable_fields

    def format(self) -> str:
        changes = SyncSummary.format_change_counts(
            updated=self.updated,
            unchanged=self.unchanged,
            skipped=self.skipped,
            errors=self.errors,
        )
        base = (
            f"Task '{self.task_name}' ({self.model}): {self.eligible} notes - {changes}"
        )
        if self.canceled:
            return f"{base}, {self.canceled} canceled"
        return base

    def format_usage(self) -> str:
        request_label = "request" if self.requests == 1 else "requests"
        provider_seconds = self.provider_latency_ms_total / 1000
        return (
            f"{self.requests} {request_label}, "
            f"{self.input_tokens} input tokens, "
            f"{self.output_tokens} output tokens, "
            f"{provider_seconds:.1f}s provider time"
        )

    def format_cost(self) -> str:
        estimate = self.model.estimate_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        if estimate is None:
            return "n/a"
        return estimate.format()


@dataclass(frozen=True)
class TaskCatalog:
    tasks_by_name: dict[str, TaskConfig]
    errors: dict[str, str]


@dataclass(frozen=True)
class LlmJobResult:
    job_id: int
    status: str
    summary: TaskRunSummary
    failed: bool
    persisted: bool


@dataclass(frozen=True)
class PlanFieldSurface:
    note_type: str
    candidate_notes: int
    editable_fields: list[str]
    read_only_fields: list[str]
    hidden_fields: list[str]
    tag_access: FieldAccess


@dataclass(frozen=True)
class TaskPlanResult:
    task_name: str
    model: ModelSpec
    deck_scope: str
    serializer_scope: str
    system_prompt_path: str | None
    user_prompt_path: str | None
    system_prompt: str
    user_prompt: str
    request_defaults: str
    summary: TaskRunSummary
    field_surface: list[PlanFieldSurface]
    requests_estimate: int
    input_tokens_estimate: int

    def format_full_prompt(self) -> str:
        return (
            f"<system>\n{self.system_prompt.strip()}\n</system>\n\n"
            f"<user>\n{self.user_prompt.strip()}\n</user>"
        )

    def format_cost_estimate(self) -> str:
        estimate = self.model.estimate_cost(
            input_tokens=self.input_tokens_estimate,
            # we assume input and output tokens are the same for estimation purposes
            output_tokens=self.input_tokens_estimate,
        )
        if estimate is None:
            return "n/a"
        return estimate.format()


@dataclass(frozen=True)
class TaskExecutionProgress:
    job_id: int
    task_name: str
    total: int
    completed: int
    in_flight: int
    queued: int
    updated: int
    unchanged: int
    skipped: int
    errors: int
    canceled: int

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 1.0
        return min(self.completed / self.total, 1.0)

    @property
    def is_finished(self) -> bool:
        return self.completed >= self.total


def _matches_any(patterns: list[str], value: str) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)

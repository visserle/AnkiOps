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


class RunFailurePolicy(Enum):
    ATOMIC = "atomic"
    PARTIAL = "partial"


class LlmCandidateStatus(Enum):
    ELIGIBLE = "eligible"
    SKIPPED_DECK_SCOPE = "skipped_deck_scope"
    SKIPPED_NO_EDITABLE_FIELDS = "skipped_no_editable_fields"
    INVALID_NOTE = "invalid_note"


class LlmFinalStatus(Enum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED_UPDATED = "succeeded_updated"
    SUCCEEDED_UNCHANGED = "succeeded_unchanged"
    NOTE_ERROR = "note_error"
    PROVIDER_ERROR = "provider_error"
    FATAL_ERROR = "fatal_error"
    CANCELED = "canceled"
    EXPIRED = "expired"


class LlmAttemptResultType(Enum):
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


class LlmJobStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
    retries: int = 2
    retry_backoff_seconds: float = 0.5
    retry_backoff_jitter: bool = True


@dataclass(frozen=True)
class TaskConfig:
    name: str
    model: AnthropicModel
    system_prompt: str
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
class PreparedAttemptRequest:
    note_payload: NotePayload
    system_prompt_text: str
    user_message_text: str
    request_params: dict[str, object]
    output_schema: dict[str, object]
    editable_fields: frozenset[str]


@dataclass(frozen=True)
class ProviderAttemptErrorContext:
    provider_message_id: str | None
    provider_model: str | None
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retry_count: int
    response_raw_text: str | None
    response_full_json: str | None


@dataclass(frozen=True)
class ProviderAttemptOutcome:
    update: NoteUpdate
    provider_message_id: str | None
    provider_model: str | None
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    retry_count: int
    response_raw_text: str
    response_full_json: str | None = None


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
    provider_retries: int = 0

    @property
    def skipped(self) -> int:
        return self.skipped_deck_scope + self.skipped_no_editable_fields

    def format(self) -> str:
        return (
            f"Task '{self.task_name}' ({self.model}): {self.eligible} notes — "
            + format_changes(
                updated=self.updated,
                unchanged=self.unchanged,
                skipped=self.skipped,
                errors=self.errors,
            )
        )

    def format_usage(self) -> str:
        request_label = "request" if self.requests == 1 else "requests"
        retry_label = "retry" if self.provider_retries == 1 else "retries"
        provider_seconds = self.provider_latency_ms_total / 1000
        return (
            f"{self.requests} {request_label}, "
            f"{self.input_tokens} input tokens, "
            f"{self.output_tokens} output tokens, "
            f"{self.provider_retries} {retry_label}, "
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


@dataclass(frozen=True)
class TaskPlanResult:
    task_name: str
    model: AnthropicModel
    deck_scope: str
    serializer_scope: str
    request_defaults: str
    summary: TaskRunSummary
    field_surface: list[PlanFieldSurface]
    requests_estimate: int
    input_tokens_estimate: int
    output_tokens_cap: int

    def format_cost_estimate(self) -> str:
        estimate = self.model.estimate_cost(
            input_tokens=self.input_tokens_estimate,
            output_tokens=self.output_tokens_cap,
        )
        return estimate.format()

"""Normalized provider outcomes for runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .payloads import NoteUpdate


class ProviderOutcomeKind(Enum):
    SUCCESS = "success"
    REFUSAL = "refusal"
    PROVIDER_ERROR = "provider_error"
    VALIDATION_ERROR = "validation_error"
    FATAL_ERROR = "fatal"


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ProviderOutcome:
    kind: ProviderOutcomeKind
    update: NoteUpdate | None = None
    refusal_text: str | None = None
    error_message: str | None = None
    provider_message_id: str | None = None
    response_model_id: str | None = None
    request_id: str | None = None
    stop_reason: str | None = None
    rate_limit_headers: dict[str, str] = field(default_factory=dict)
    usage: ProviderUsage = ProviderUsage()
    latency_ms: int = 0
    retry_count: int = 0
    raw_text: str | None = None
    raw_json: str | None = None

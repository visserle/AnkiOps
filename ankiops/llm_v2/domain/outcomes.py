"""Normalized provider outcomes for runtime v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .payloads import NoteUpdate


class ProviderOutcomeKind(Enum):
    SUCCESS = "success"
    REFUSAL = "refusal"
    PROVIDER_ERROR = "provider_error"
    VALIDATION_ERROR = "validation_error"
    FATAL_ERROR = "fatal_error"


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
    usage: ProviderUsage = ProviderUsage()
    latency_ms: int = 0
    raw_text: str | None = None
    raw_json: str | None = None

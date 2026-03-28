"""Shared error types for Claude task execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_models import ProviderAttemptErrorContext


class LlmFatalError(RuntimeError):
    """Raised for fatal Claude API failures that should abort the run."""


class LlmNoteError(RuntimeError):
    """Raised for note-scoped Claude failures."""

    def __init__(
        self,
        message: str,
        *,
        attempt_context: ProviderAttemptErrorContext | None = None,
    ) -> None:
        super().__init__(message)
        self.attempt_context = attempt_context

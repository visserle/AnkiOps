"""Shared error types for Claude task execution."""


class LlmFatalError(RuntimeError):
    """Raised for fatal Claude API failures that should abort the run."""


class LlmNoteError(RuntimeError):
    """Raised for note-scoped Claude failures."""

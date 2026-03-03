"""Shared error types for LLM providers."""


class ProviderFatalError(RuntimeError):
    """Raised for fatal provider failures that should abort the run."""


class ProviderNoteError(RuntimeError):
    """Raised for note-scoped provider failures."""

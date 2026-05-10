"""Error types for LLM runtime."""

from __future__ import annotations


class LlmV2Error(RuntimeError):
    """Base exception for runtime failures."""


class CapabilityError(LlmV2Error):
    """Raised when model capabilities are invalid for execution."""


class ContractValidationError(LlmV2Error, ValueError):
    """Raised when structured output contract validation fails."""


class RuntimeFatalError(LlmV2Error):
    """Raised when a runtime error should abort the whole LLM job."""

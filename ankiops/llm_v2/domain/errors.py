"""Error types for LLM runtime v2."""

from __future__ import annotations


class LlmV2Error(RuntimeError):
    """Base exception for runtime v2 failures."""


class CapabilityError(LlmV2Error):
    """Raised when model capabilities are invalid for execution."""


class ContractValidationError(LlmV2Error, ValueError):
    """Raised when structured output contract validation fails."""

"""Domain exceptions for the AI module."""

from __future__ import annotations


class AIError(Exception):
    """Base exception for all AI module failures."""


class AIConfigError(AIError):
    """Invalid AI model/runtime configuration."""


class TaskConfigError(AIError):
    """Invalid task configuration or task reference."""


class AIRequestError(AIError):
    """AI request failed after retry policy or due to request validation."""


class AIResponseError(AIError):
    """AI returned an invalid or unsupported response payload."""


class TaskExecutionError(AIError):
    """Task execution could not proceed due to invalid run input/state."""

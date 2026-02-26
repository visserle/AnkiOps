"""Runtime transport selection for AI editors."""

from __future__ import annotations

from .client import OpenAICompatibleAsyncEditor
from .errors import AIConfigError
from .types import AsyncInlineBatchEditor, RuntimeAIConfig


def build_async_editor(config: RuntimeAIConfig) -> AsyncInlineBatchEditor:
    """Construct the async editor implementation for a runtime config."""
    if config.transport == "openai_chat_completions":
        return OpenAICompatibleAsyncEditor(config)
    raise AIConfigError(
        f"Unsupported AI transport '{config.transport}' for provider '{config.provider}'"
    )

"""Runtime transport selection for AI editors."""

from __future__ import annotations

from ankiops.ai.client import OpenAICompatibleAsyncEditor
from ankiops.ai.errors import AIConfigError
from ankiops.ai.types import AsyncInlineBatchEditor, RuntimeAIConfig


def build_async_editor(config: RuntimeAIConfig) -> AsyncInlineBatchEditor:
    """Construct the async editor implementation for a runtime config."""
    if config.transport == "openai_chat_completions":
        return OpenAICompatibleAsyncEditor(config)
    raise AIConfigError(
        "Unsupported AI transport "
        f"'{config.transport}' for provider '{config.provider}'"
    )

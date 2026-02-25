"""Public AI API for prompt-driven inline JSON editing."""

from .client import OpenAICompatibleAsyncEditor
from .errors import (
    AIConfigError,
    AIError,
    AIRequestError,
    AIResponseError,
    PromptConfigError,
    PromptExecutionError,
)
from .model_profiles import (
    MODELS_FILE_NAME,
    load_model_profiles,
    resolve_runtime_config,
)
from .prompts import load_prompt, resolve_prompt_path
from .runner import PromptRunner
from .types import (
    AsyncInlineBatchEditor,
    InlineEditedNote,
    InlineNotePayload,
    ModelProfile,
    ModelsConfig,
    PromptChange,
    PromptConfig,
    PromptRunOptions,
    PromptRunResult,
    RuntimeAIConfig,
)

__all__ = [
    "AsyncInlineBatchEditor",
    "AIConfigError",
    "AIError",
    "AIRequestError",
    "AIResponseError",
    "InlineEditedNote",
    "InlineNotePayload",
    "MODELS_FILE_NAME",
    "ModelProfile",
    "ModelsConfig",
    "OpenAICompatibleAsyncEditor",
    "PromptChange",
    "PromptConfigError",
    "PromptConfig",
    "PromptExecutionError",
    "PromptRunOptions",
    "PromptRunResult",
    "PromptRunner",
    "RuntimeAIConfig",
    "load_model_profiles",
    "load_prompt",
    "resolve_prompt_path",
    "resolve_runtime_config",
]

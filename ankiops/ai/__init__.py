"""Public AI API for prompt-driven inline JSON editing."""

from .client import OpenAICompatibleAsyncEditor
from .model_profiles import (
    MODELS_FILE_NAME,
    load_models_config,
    models_config_path,
    resolve_runtime_ai_config,
)
from .prompts import load_prompt_config, resolve_prompt_path
from .runner import (
    run_inline_prompt_on_serialized_collection,
    run_inline_prompt_on_serialized_collection_async,
    select_decks_with_subdecks,
)
from .types import (
    AsyncInlineBatchEditor,
    ModelProfile,
    ModelsConfig,
    PromptChange,
    PromptConfig,
    PromptRunResult,
    RuntimeAIConfig,
)

__all__ = [
    "AsyncInlineBatchEditor",
    "MODELS_FILE_NAME",
    "ModelProfile",
    "ModelsConfig",
    "OpenAICompatibleAsyncEditor",
    "PromptChange",
    "PromptConfig",
    "PromptRunResult",
    "RuntimeAIConfig",
    "load_models_config",
    "load_prompt_config",
    "models_config_path",
    "resolve_prompt_path",
    "resolve_runtime_ai_config",
    "run_inline_prompt_on_serialized_collection",
    "run_inline_prompt_on_serialized_collection_async",
    "select_decks_with_subdecks",
]

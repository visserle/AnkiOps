"""Public AI API for task-driven inline JSON editing."""

from ankiops.ai.client import OpenAICompatibleAsyncEditor
from ankiops.ai.errors import (
    AIConfigError,
    AIError,
    AIRequestError,
    AIResponseError,
    TaskConfigError,
    TaskExecutionError,
)
from ankiops.ai.orchestration import AIRuntimeOverrides, prepare_ai_run
from ankiops.ai.paths import AIPaths
from ankiops.ai.runner import TaskRunner
from ankiops.ai.transports import build_async_editor
from ankiops.ai.types import TaskRunOptions

__all__ = [
    "AIConfigError",
    "AIError",
    "AIPaths",
    "AIRequestError",
    "AIResponseError",
    "AIRuntimeOverrides",
    "OpenAICompatibleAsyncEditor",
    "TaskConfigError",
    "TaskExecutionError",
    "TaskRunOptions",
    "TaskRunner",
    "build_async_editor",
    "prepare_ai_run",
]

"""Public AI API for task-driven inline JSON editing."""

from .client import OpenAICompatibleAsyncEditor
from .errors import (
    AIConfigError,
    AIError,
    AIRequestError,
    AIResponseError,
    TaskConfigError,
    TaskExecutionError,
)
from .orchestration import AIRuntimeOverrides, prepare_ai_run
from .paths import AIPaths
from .runner import TaskRunner
from .transports import build_async_editor
from .types import TaskRunOptions

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

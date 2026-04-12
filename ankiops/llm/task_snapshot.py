"""Serialization helpers for persisted task snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .anthropic_models import parse_model
from .llm_models import (
    DeckScope,
    ExecutionMode,
    FieldExceptionRule,
    TaskConfig,
    TaskExecutionOptions,
    TaskRequestOptions,
)

_DEFAULT_REQUEST_OPTIONS = TaskRequestOptions()
_DEFAULT_EXECUTION_OPTIONS = TaskExecutionOptions()
_DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
_DEFAULT_TIMEOUT_SECONDS = 60


def task_to_snapshot(task: TaskConfig) -> dict[str, Any]:
    return {
        "name": task.name,
        "model": task.model.name,
        "system_prompt": task.system_prompt,
        "prompt": task.prompt,
        "system_prompt_path": str(task.system_prompt_path),
        "prompt_path": str(task.prompt_path),
        "api_key_env": task.api_key_env,
        "timeout_seconds": task.timeout_seconds,
        "decks": {
            "deck_root": task.decks.deck_root,
        },
        "field_exceptions": [
            {
                "note_types": list(rule.note_types),
                "read_only": list(rule.read_only),
                "hidden": list(rule.hidden),
            }
            for rule in task.field_exceptions
        ],
        "request": {
            "temperature": task.request.temperature,
            "max_output_tokens": task.request.max_output_tokens,
            "retries": task.request.retries,
            "retry_backoff_seconds": task.request.retry_backoff_seconds,
            "retry_backoff_jitter": task.request.retry_backoff_jitter,
        },
        "execution": {
            "mode": task.execution.mode.value,
            "concurrency": task.execution.concurrency,
            "batch_poll_seconds": task.execution.batch_poll_seconds,
        },
    }


def task_from_snapshot(snapshot: dict[str, Any]) -> TaskConfig:
    model_name = snapshot.get("model")
    if not isinstance(model_name, str):
        raise ValueError("Job snapshot is missing model")
    model = parse_model(model_name)
    if model is None:
        raise ValueError(f"Job snapshot references unsupported model '{model_name}'")

    decks = _require_mapping(snapshot, "decks", "Job snapshot is missing deck scope")
    request = _require_mapping(
        snapshot,
        "request",
        "Job snapshot is missing request options",
    )
    execution = _require_mapping(
        snapshot,
        "execution",
        "Job snapshot is missing execution options",
    )

    mode_raw = execution.get("mode")
    if not isinstance(mode_raw, str):
        raise ValueError("Job snapshot execution mode is invalid")

    deck_root_raw = decks.get("deck_root")
    if deck_root_raw is not None and not isinstance(deck_root_raw, str):
        raise ValueError("Job snapshot deck scope is invalid")
    deck_root = deck_root_raw.strip() if isinstance(deck_root_raw, str) else None
    if deck_root == "":
        deck_root = None

    field_exceptions: list[FieldExceptionRule] = []
    field_exceptions_raw = snapshot.get("field_exceptions")
    if isinstance(field_exceptions_raw, list):
        for entry in field_exceptions_raw:
            if not isinstance(entry, dict):
                continue
            field_exceptions.append(
                FieldExceptionRule(
                    note_types=_string_list(entry.get("note_types"), default=["*"]),
                    read_only=_string_list(entry.get("read_only"), default=[]),
                    hidden=_string_list(entry.get("hidden"), default=[]),
                )
            )

    return TaskConfig(
        name=str(snapshot.get("name") or "unknown"),
        model=model,
        system_prompt=str(snapshot.get("system_prompt") or ""),
        prompt=str(snapshot.get("prompt") or ""),
        system_prompt_path=Path(str(snapshot.get("system_prompt_path") or "")),
        prompt_path=Path(str(snapshot.get("prompt_path") or "")),
        api_key_env=str(snapshot.get("api_key_env") or _DEFAULT_API_KEY_ENV),
        timeout_seconds=int(
            snapshot.get("timeout_seconds") or _DEFAULT_TIMEOUT_SECONDS
        ),
        decks=DeckScope(deck_root=deck_root),
        field_exceptions=field_exceptions,
        request=TaskRequestOptions(
            temperature=(
                float(request["temperature"])
                if request.get("temperature") is not None
                else None
            ),
            max_output_tokens=(
                int(request["max_output_tokens"])
                if request.get("max_output_tokens") is not None
                else None
            ),
            retries=int(request.get("retries", _DEFAULT_REQUEST_OPTIONS.retries)),
            retry_backoff_seconds=float(
                request.get(
                    "retry_backoff_seconds",
                    _DEFAULT_REQUEST_OPTIONS.retry_backoff_seconds,
                )
            ),
            retry_backoff_jitter=bool(
                request.get(
                    "retry_backoff_jitter",
                    _DEFAULT_REQUEST_OPTIONS.retry_backoff_jitter,
                )
            ),
        ),
        execution=TaskExecutionOptions(
            mode=ExecutionMode(mode_raw),
            concurrency=int(
                execution.get("concurrency", _DEFAULT_EXECUTION_OPTIONS.concurrency)
            ),
            batch_poll_seconds=int(
                execution.get(
                    "batch_poll_seconds",
                    _DEFAULT_EXECUTION_OPTIONS.batch_poll_seconds,
                )
            ),
        ),
    )


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    error_message: str,
) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(error_message)
    return value


def _string_list(value: object, *, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    return [item for item in value if isinstance(item, str)]

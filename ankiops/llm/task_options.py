"""Task option normalization and formatting helpers."""

from __future__ import annotations

from dataclasses import replace

from .anthropic_models import (
    AnthropicModel,
    format_supported_model_names,
    parse_model,
)
from .llm_models import DeckScope, ExecutionMode, RunFailurePolicy, TaskConfig


def resolve_serializer_scope(task: TaskConfig) -> tuple[str | None, bool]:
    include = task.decks.include
    if task.decks.exclude or len(include) != 1:
        return None, False

    deck_name = include[0].strip()
    if not deck_name:
        return None, False

    if any(char in deck_name for char in ("*", "?", "[")):
        return None, False

    return deck_name, not task.decks.include_subdecks


def apply_deck_override(task: TaskConfig, deck_override: str | None) -> TaskConfig:
    if deck_override is None:
        return task

    deck_name = deck_override.strip()
    if not deck_name:
        raise ValueError("Deck override must be a non-empty deck name")
    if any(char in deck_name for char in ("*", "?", "[")):
        raise ValueError(
            "Deck override must be an exact deck name (wildcards are not supported)"
        )

    return replace(
        task,
        decks=DeckScope(
            include=[deck_name],
            exclude=[],
            include_subdecks=False,
        ),
    )


def apply_mode_override(task: TaskConfig, mode_override: str | None) -> TaskConfig:
    if mode_override is None:
        return task

    normalized = mode_override.strip().lower()
    try:
        execution_mode = ExecutionMode(normalized)
    except ValueError as error:
        supported = ", ".join(mode.value for mode in ExecutionMode)
        raise ValueError(f"Execution mode must be one of: {supported}") from error

    return replace(
        task,
        execution=replace(task.execution, mode=execution_mode),
    )


def format_deck_scope(task: TaskConfig) -> str:
    include = ",".join(task.decks.include)
    exclude = ",".join(task.decks.exclude) if task.decks.exclude else "-"
    if include == "*" and exclude == "-" and task.decks.include_subdecks:
        return "*"
    return (
        f"include={include} exclude={exclude} "
        f"include_subdecks={str(task.decks.include_subdecks).lower()}"
    )


def format_serializer_scope(deck: str | None, no_subdecks: bool) -> str:
    if deck is None:
        return "*"
    if no_subdecks:
        return f"exact:{deck}"
    return deck


def format_request_defaults(task: TaskConfig) -> str:
    max_tokens = task.request.max_output_tokens or 2048
    temperature = (
        task.request.temperature if task.request.temperature is not None else "default"
    )
    execution = task.execution
    if execution.mode is ExecutionMode.ONLINE:
        execution_text = (
            f"mode=online concurrency={execution.concurrency} "
            f"fail_fast={str(execution.fail_fast).lower()}"
        )
    else:
        execution_text = (
            f"mode=batch poll={execution.batch_poll_seconds}s "
            f"fail_fast={str(execution.fail_fast).lower()}"
        )

    return (
        f"timeout={task.timeout_seconds}s "
        f"max_tokens={max_tokens} temperature={temperature} "
        f"retries={task.request.retries} "
        f"retry_backoff={task.request.retry_backoff_seconds}s "
        f"retry_jitter={str(task.request.retry_backoff_jitter).lower()} "
        f"{execution_text}"
    )


def resolve_model(
    task: TaskConfig,
    model_override: str | None,
) -> AnthropicModel:
    if model_override is None:
        return task.model

    model = parse_model(model_override)
    if model is None:
        supported = format_supported_model_names()
        raise ValueError(f"Model must be one of: {supported}")
    return model


def resolve_failure_policy(
    value: RunFailurePolicy | str,
) -> RunFailurePolicy:
    if isinstance(value, RunFailurePolicy):
        return value

    normalized = value.strip().lower()
    for policy in RunFailurePolicy:
        if policy.value == normalized:
            return policy
    supported = ", ".join(policy.value for policy in RunFailurePolicy)
    raise ValueError(f"Failure policy must be one of: {supported}")

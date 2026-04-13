"""Task option normalization and formatting helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .llm_models import DeckScope, ExecutionMode, RunFailurePolicy, TaskConfig
from .model_registry import (
    ProviderModel,
    format_supported_model_names,
    parse_model,
)


def resolve_serializer_scope(task: TaskConfig) -> tuple[str | None, bool]:
    if task.decks.deck_root is None:
        return None, False
    return task.decks.deck_root, False


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
        decks=DeckScope(deck_root=deck_name),
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
    if task.decks.deck_root is None:
        return "collection"
    return f"deck:{task.decks.deck_root}"


def format_request_defaults(task: TaskConfig) -> str:
    max_tokens = task.request.max_output_tokens or 2048
    temperature = (
        task.request.temperature if task.request.temperature is not None else "default"
    )
    execution_text = f"mode=online concurrency={task.execution.concurrency}"

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
    *,
    collection_dir: Path,
) -> ProviderModel:
    if model_override is None:
        return task.model

    model = parse_model(model_override, collection_dir=collection_dir)
    if model is None:
        supported = format_supported_model_names(collection_dir=collection_dir)
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

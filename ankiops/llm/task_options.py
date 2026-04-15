"""Task option normalization and formatting helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .model_registry import ModelSpec, load_model_registry
from .task_types import DeckScope, TaskConfig


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

    return replace(task, decks=DeckScope(deck_root=deck_name))


def format_deck_scope(task: TaskConfig) -> str:
    return (
        "collection" if task.decks.deck_root is None else f"deck:{task.decks.deck_root}"
    )


def format_request_defaults(task: TaskConfig) -> str:
    max_tokens = task.request.max_output_tokens
    temperature = (
        task.request.temperature if task.request.temperature is not None else "default"
    )
    execution_text = f"concurrency={task.model.concurrency}"

    return (
        f"timeout={task.timeout_seconds}s "
        f"max_tokens={max_tokens} temperature={temperature} "
        f"retries={task.model.retries} "
        f"retry_backoff={task.model.retry_backoff_seconds}s "
        f"retry_jitter={str(task.model.retry_backoff_jitter).lower()} "
        f"{execution_text}"
    )


def resolve_model(
    task: TaskConfig,
    model_override: str | None,
    *,
    collection_dir: Path,
) -> ModelSpec:
    if model_override is None:
        return task.model

    registry = load_model_registry(collection_dir=collection_dir)
    model = registry.parse(model_override)
    if model is None:
        supported = registry.format_models()
        raise ValueError(f"Model must be one of: {supported}")
    return model

"""Execution of collection-local Claude tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.models import ANKIOPS_KEY_FIELD, Note, NoteTypeConfig

from .anthropic_models import format_supported_model_names, parse_model
from .claude import ClaudeClient
from .config_loader import load_llm_task_catalog
from .errors import LlmFatalError, LlmNoteError
from .models import (
    FieldAccess,
    GenerateUpdateResult,
    NotePayload,
    RunFailurePolicy,
    TaskConfig,
    TaskRunResult,
    TaskRunSummary,
)

logger = logging.getLogger(__name__)


def _resolve_serializer_scope(task: TaskConfig) -> tuple[str | None, bool]:
    include = task.decks.include
    if task.decks.exclude or len(include) != 1:
        return None, False

    deck_name = include[0].strip()
    if not deck_name:
        return None, False

    if any(char in deck_name for char in ("*", "?", "[")):
        return None, False

    return deck_name, not task.decks.include_subdecks


def _format_deck_scope(task: TaskConfig) -> str:
    include = ",".join(task.decks.include)
    exclude = ",".join(task.decks.exclude) if task.decks.exclude else "-"
    if include == "*" and exclude == "-" and task.decks.include_subdecks:
        return "*"
    return (
        f"include={include} exclude={exclude} "
        f"include_subdecks={str(task.decks.include_subdecks).lower()}"
    )


def _format_serializer_scope(deck: str | None, no_subdecks: bool) -> str:
    if deck is None:
        return "*"
    if no_subdecks:
        return f"exact:{deck}"
    return deck


def _format_request_defaults(task: TaskConfig) -> str:
    max_tokens = task.request.max_output_tokens or 2048
    temperature = (
        task.request.temperature if task.request.temperature is not None else "default"
    )
    return (
        f"timeout={task.timeout_seconds}s "
        f"max_tokens={max_tokens} temperature={temperature} "
        f"retries={task.request.retries} "
        f"retry_backoff={task.request.retry_backoff_seconds}s "
        f"retry_jitter={str(task.request.retry_backoff_jitter).lower()}"
    )


def _resolve_model(task: TaskConfig, model_override: str | None):
    if model_override is None:
        return task.model

    model = parse_model(model_override)
    if model is None:
        supported = format_supported_model_names()
        raise ValueError(f"Model must be one of: {supported}")
    return model


def _resolve_failure_policy(
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


def _build_note_payload(
    task: TaskConfig,
    *,
    note: dict[str, Any],
    note_type_field_names: set[str],
) -> NotePayload | None:
    note_key = note.get("note_key")
    note_type = note.get("note_type")
    fields = note.get("fields")
    if (
        not isinstance(note_key, str)
        or not isinstance(note_type, str)
        or not isinstance(fields, dict)
    ):
        raise ValueError("Serialized note is missing note_key, note_type, or fields")

    editable_fields: dict[str, str] = {}
    read_only_fields: dict[str, str] = {}

    for field_name, raw_value in fields.items():
        if field_name == ANKIOPS_KEY_FIELD.name:
            continue
        if field_name not in note_type_field_names:
            continue
        if not isinstance(raw_value, str) or not raw_value:
            continue

        access = task.field_access(note_type, field_name)
        if access is FieldAccess.HIDDEN:
            continue
        if access is FieldAccess.READ_ONLY:
            read_only_fields[field_name] = raw_value
        else:
            editable_fields[field_name] = raw_value

    if not editable_fields:
        return None

    return NotePayload(
        note_key=note_key,
        note_type=note_type,
        editable_fields=editable_fields,
        read_only_fields=read_only_fields,
    )


def _apply_note_update(
    *,
    serialized_note: dict[str, Any],
    payload: NotePayload,
    edits: dict[str, str],
    note_type_config: NoteTypeConfig,
) -> list[str]:
    if payload.note_key != serialized_note.get("note_key"):
        raise LlmNoteError("Update note_key did not match serialized note")

    raw_fields = serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise LlmNoteError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    changed_fields: list[str] = []
    for field_name, value in edits.items():
        if field_name not in payload.editable_fields:
            if field_name in payload.read_only_fields:
                raise LlmNoteError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise LlmNoteError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        if next_fields.get(field_name) != value:
            next_fields[field_name] = value
            changed_fields.append(field_name)

    if not changed_fields:
        return []

    note = Note(
        note_key=payload.note_key,
        note_type=payload.note_type,
        fields=next_fields,
    )
    errors = note.validate(note_type_config)
    if errors:
        raise LlmNoteError("; ".join(errors))

    serialized_note["fields"] = next_fields
    return changed_fields


def _load_task(
    *,
    collection_dir: Path,
    task_name: str,
) -> tuple[TaskConfig, dict[str, NoteTypeConfig]]:
    fs = FileSystemAdapter()
    note_type_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    catalog = load_llm_task_catalog(
        collection_dir,
        note_type_configs=note_type_configs,
    )
    if catalog.errors:
        joined_errors = "\n".join(catalog.errors.values())
        raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")
    task = catalog.tasks_by_name.get(task_name)
    if task is None:
        raise ValueError(f"Unknown or invalid task '{task_name}'")

    config_by_name = {config.name: config for config in note_type_configs}
    return task, config_by_name


def _iter_decks(
    data: dict[str, Any],
    *,
    summary: TaskRunSummary,
) -> Iterator[tuple[str, list[Any]]]:
    decks = data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Serialized collection is missing a decks list")

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            continue

        summary.decks_seen += 1
        summary.notes_seen += len(notes)
        yield deck_name, notes


def _record_provider_result(
    summary: TaskRunSummary,
    result: GenerateUpdateResult,
) -> None:
    summary.requests += 1
    summary.input_tokens += result.input_tokens
    summary.output_tokens += result.output_tokens
    summary.provider_latency_ms_total += result.latency_ms
    summary.provider_retries += result.retry_count


def _process_note(
    *,
    deck_name: str,
    note: dict[str, Any],
    task: TaskConfig,
    api_model: str,
    note_type_configs: dict[str, NoteTypeConfig],
    provider_client: ClaudeClient,
    summary: TaskRunSummary,
) -> None:
    note_key = note.get("note_key")
    note_type_name = note.get("note_type")
    note_label = note_key if isinstance(note_key, str) else "unknown"
    note_type_label = note_type_name if isinstance(note_type_name, str) else "unknown"

    note_type_config = (
        note_type_configs.get(note_type_name)
        if isinstance(note_type_name, str)
        else None
    )
    if note_type_config is None:
        raise ValueError(f"Unknown note type '{note_type_name}' in serialized note")

    note_field_names = {
        field.name
        for field in note_type_config.fields
        if field.name != ANKIOPS_KEY_FIELD.name
    }
    payload = _build_note_payload(
        task,
        note=note,
        note_type_field_names=note_field_names,
    )
    if payload is None:
        summary.skipped_no_editable_fields += 1
        logger.debug(
            "  Skipped %s in '%s' (%s): no editable non-empty fields",
            note_label,
            deck_name,
            note_type_label,
        )
        return

    summary.eligible += 1
    try:
        result = provider_client.generate_update(
            note_payload=payload,
            task_prompt=task.prompt,
            request_options=task.request,
            api_model=api_model,
        )
        _record_provider_result(summary, result)

        update = result.update
        if update.note_key != payload.note_key:
            raise LlmNoteError("Model returned mismatched note_key")

        changed_fields = _apply_note_update(
            serialized_note=note,
            payload=payload,
            edits=update.edits,
            note_type_config=note_type_config,
        )
    except LlmFatalError:
        raise
    except LlmNoteError as error:
        summary.errors += 1
        logger.error(
            "LLM note error for %s in '%s' (%s): %s",
            payload.note_key,
            deck_name,
            payload.note_type,
            error,
        )
        return

    if changed_fields:
        summary.updated += 1
        logger.debug(
            "  Updated %s in '%s' (%s): fields=%s",
            payload.note_key,
            deck_name,
            payload.note_type,
            ",".join(changed_fields),
        )
        return

    summary.unchanged += 1
    logger.debug(
        "  Unchanged %s in '%s' (%s)",
        payload.note_key,
        deck_name,
        payload.note_type,
    )


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> TaskRunResult:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )

    model = _resolve_model(task, model_override)
    resolved_failure_policy = _resolve_failure_policy(failure_policy)

    provider_client = ClaudeClient(task)
    summary = TaskRunSummary(
        task_name=task.name,
        model=model,
    )

    deck, no_subdecks = _resolve_serializer_scope(task)
    logger.debug(
        "Starting LLM task '%s' (model=%s, api_model=%s, collection=%s, "
        "deck_scope=%s, failure_policy=%s)",
        task.name,
        model,
        model.api_id,
        collection_dir,
        _format_deck_scope(task),
        resolved_failure_policy.value,
    )
    logger.debug("LLM request defaults: %s", _format_request_defaults(task))
    logger.debug(
        "LLM serializer scope: %s",
        _format_serializer_scope(deck, no_subdecks),
    )

    if not no_auto_commit:
        logger.debug("Creating pre-LLM git snapshot")
        git_snapshot(collection_dir, f"llm:{task.name}")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    data = serialize_collection(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )

    for deck_name, notes in _iter_decks(data, summary=summary):
        if task.decks.matches(deck_name):
            summary.decks_matched += 1
        else:
            summary.skipped_deck_scope += len(notes)
            logger.debug(
                "Skipping deck '%s' (%d notes): outside task scope",
                deck_name,
                len(notes),
            )
            continue

        for note in notes:
            if not isinstance(note, dict):
                continue
            _process_note(
                deck_name=deck_name,
                note=note,
                task=task,
                api_model=model.api_id,
                note_type_configs=note_type_configs,
                provider_client=provider_client,
                summary=summary,
            )

    persisted = False
    if summary.updated > 0:
        if resolved_failure_policy is RunFailurePolicy.ATOMIC and summary.errors:
            logger.error(
                "Atomic failure policy prevented persistence: %d update(s) staged, "
                "%d error(s) observed",
                summary.updated,
                summary.errors,
            )
        else:
            deserialize_collection_data(
                data,
                root_dir=collection_dir,
                note_types_dir=collection_dir / NOTE_TYPES_DIR,
                overwrite=True,
            )
            persisted = True

    logger.info(summary.format())
    logger.debug(summary.format_usage())
    logger.debug(summary.format_cost())
    return TaskRunResult(
        summary=summary,
        failed=summary.errors > 0,
        persisted=persisted,
    )

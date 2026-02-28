"""Execution of collection-local LLM tasks."""

from __future__ import annotations

import logging
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

from .config_loader import load_llm_config_set
from .models import (
    FieldAccess,
    LlmConfigSet,
    NotePayload,
    ProviderConfig,
    ProviderType,
    TaskConfig,
    TaskRunSummary,
)
from .prompting import build_instructions
from .providers import (
    LlmProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderFatalError,
    ProviderNoteError,
)

logger = logging.getLogger(__name__)


def _provider_for(config: ProviderConfig) -> LlmProvider:
    if config.type is ProviderType.OPENAI:
        return OpenAIProvider(config)
    if config.type is ProviderType.OLLAMA:
        return OllamaProvider(config)
    raise ValueError(f"Unsupported provider type: {config.type.value}")


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


def _apply_note_patch(
    *,
    serialized_note: dict[str, Any],
    payload: NotePayload,
    edits: dict[str, str],
    note_type_config,
) -> bool:
    if payload.note_key != serialized_note.get("note_key"):
        raise ProviderNoteError("Patch note_key did not match serialized note")

    raw_fields = serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise ProviderNoteError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    changed = False
    for field_name, value in edits.items():
        if field_name not in payload.editable_fields:
            if field_name in payload.read_only_fields:
                raise ProviderNoteError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise ProviderNoteError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        if next_fields.get(field_name) != value:
            next_fields[field_name] = value
            changed = True

    if not changed:
        return False

    note = Note(
        note_key=payload.note_key,
        note_type=payload.note_type,
        fields=next_fields,
    )
    errors = note.validate(note_type_config)
    if errors:
        raise ProviderNoteError("; ".join(errors))

    serialized_note["fields"] = next_fields
    return True


def _load_config_set(
    collection_dir: Path,
) -> tuple[LlmConfigSet, dict[str, NoteTypeConfig]]:
    fs = FileSystemAdapter()
    note_type_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    config_set = load_llm_config_set(
        collection_dir,
        note_type_configs=note_type_configs,
    )
    config_by_name = {config.name: config for config in note_type_configs}
    return config_set, config_by_name


def list_tasks(collection_dir: Path) -> tuple[list[TaskConfig], dict[str, str]]:
    config_set, _ = _load_config_set(collection_dir)
    errors = {**config_set.provider_errors, **config_set.task_errors}
    return list(config_set.tasks_by_name.values()), errors


def list_providers(collection_dir: Path) -> tuple[list[ProviderConfig], dict[str, str]]:
    config_set, _ = _load_config_set(collection_dir)
    errors = {**config_set.provider_errors}
    return list(config_set.providers_by_name.values()), errors


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    provider_override: str | None = None,
    model_override: str | None = None,
    dry_run: bool = False,
    no_auto_commit: bool = False,
) -> TaskRunSummary:
    config_set, note_type_configs = _load_config_set(collection_dir)

    task = config_set.tasks_by_name.get(task_name)
    if task is None:
        raise ValueError(f"Unknown or invalid task '{task_name}'")

    provider_name = provider_override or task.provider
    provider = config_set.providers_by_name.get(provider_name)
    if provider is None:
        raise ValueError(f"Unknown or invalid provider '{provider_name}'")

    model = model_override or provider.model
    request_options = provider.request_defaults.merged(task.request)
    provider_client = _provider_for(provider)
    summary = TaskRunSummary(
        task_name=task.name,
        provider_name=provider.name,
        provider_type=provider.type,
        model=model,
    )

    if not no_auto_commit:
        git_snapshot(collection_dir, f"llm:{task.name}")

    data = serialize_collection(collection_dir, strict=True)
    instructions = build_instructions(task.prompt)

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

        if not task.decks.matches(deck_name):
            summary.skipped += len(notes)
            continue

        for note in notes:
            if not isinstance(note, dict):
                continue

            note_type_name = note.get("note_type")
            note_type_config = (
                note_type_configs.get(note_type_name)
                if isinstance(note_type_name, str)
                else None
            )
            if note_type_config is None:
                raise ValueError(
                    f"Unknown note type '{note_type_name}' in serialized note"
                )

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
                summary.skipped += 1
                continue

            summary.eligible += 1
            try:
                patch = provider_client.generate_patch(
                    note_payload=payload,
                    instructions=instructions,
                    request_options=request_options,
                    model=model,
                )
                if patch.note_key != payload.note_key:
                    raise ProviderNoteError("Model returned mismatched note_key")
                changed = _apply_note_patch(
                    serialized_note=note,
                    payload=payload,
                    edits=patch.edits,
                    note_type_config=note_type_config,
                )
            except ProviderFatalError:
                raise
            except ProviderNoteError as error:
                summary.errors += 1
                logger.warning(
                    f"LLM note error in {deck_name} ({payload.note_key}): {error}"
                )
                continue

            if changed:
                summary.updated += 1
            else:
                summary.unchanged += 1

    if dry_run:
        logger.info("Dry run: no files written")
    elif summary.updated > 0:
        deserialize_collection_data(data, overwrite=True)

    logger.info(summary.format())
    if summary.errors:
        raise SystemExit(1)
    return summary

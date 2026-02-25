"""Prompt runner with async batch execution and deterministic application."""

from __future__ import annotations

import asyncio
import copy
import fnmatch
from dataclasses import dataclass
from typing import Any

from .types import (
    AsyncInlineBatchEditor,
    PromptChange,
    PromptConfig,
    PromptRunResult,
)


@dataclass
class _NoteTask:
    deck_name: str
    note_key: str
    note_type: str
    note_fields: dict[str, Any]
    string_fields: dict[str, str]
    target_field_names: list[str]
    send_field_names: list[str]
    note_payload: dict[str, Any]
    deck_ref: dict[str, Any]


@dataclass(frozen=True)
class _ChunkResponse:
    index: int
    chunk: list[_NoteTask]
    edited_by_note_key: dict[str, dict[str, Any]] | None
    error: str | None


def select_decks_with_subdecks(
    decks: list[dict[str, Any]],
    include_decks: list[str] | None,
) -> list[dict[str, Any]]:
    """Select decks by recursive include semantics."""
    targets = [
        deck_name.strip() for deck_name in (include_decks or []) if deck_name.strip()
    ]
    if not targets:
        return decks

    selected: list[dict[str, Any]] = []
    for deck in decks:
        deck_name = str(deck.get("name", ""))
        if any(_matches_deck_or_subdeck(deck_name, target) for target in targets):
            selected.append(deck)
    return selected


def _matches_deck_or_subdeck(deck_name: str, target: str) -> bool:
    return deck_name == target or deck_name.startswith(f"{target}::")


def _matching_field_names(fields: dict[str, str], patterns: list[str]) -> list[str]:
    return [
        name
        for name in fields
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]


async def run_inline_prompt_on_serialized_collection_async(
    serialized_data: dict[str, Any],
    include_decks: list[str] | None,
    prompt_config: PromptConfig,
    editor: AsyncInlineBatchEditor,
    *,
    batch_size: int = 1,
    max_in_flight: int = 4,
) -> PromptRunResult:
    """Run prompt-driven inline JSON edits over selected decks."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be > 0")

    decks = serialized_data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("serialized_data must contain 'decks' list")

    selected = select_decks_with_subdecks(decks, include_decks)
    selected_copy = copy.deepcopy(selected)
    result = PromptRunResult()
    tasks = _collect_note_tasks(selected_copy, prompt_config, result)

    chunks = _chunk_tasks(tasks, batch_size=batch_size)
    responses = await _dispatch_chunks(
        prompt_config=prompt_config,
        editor=editor,
        chunks=chunks,
        max_in_flight=max_in_flight,
    )

    changed_decks: dict[int, dict[str, Any]] = {}
    for chunk_response in responses:
        if chunk_response.error is not None:
            for task in chunk_response.chunk:
                result.warnings.append(
                    f"{task.deck_name}/{task.note_key}: {chunk_response.error}"
                )
            continue

        edited_by_note_key = chunk_response.edited_by_note_key or {}
        for task in chunk_response.chunk:
            edited = edited_by_note_key.get(task.note_key)
            if edited is None:
                result.warnings.append(
                    f"{task.deck_name}/{task.note_key}: response missing note_key"
                )
                continue

            error = _validate_edited_note(task, edited)
            if error:
                result.warnings.append(f"{task.deck_name}/{task.note_key}: {error}")
                continue

            edited_fields = edited["fields"]
            for field_name in task.target_field_names:
                edited_text = edited_fields[field_name]
                original_text = task.string_fields[field_name]
                if edited_text == original_text:
                    continue

                task.note_fields[field_name] = edited_text
                changed_decks[id(task.deck_ref)] = task.deck_ref
                result.changed_fields += 1
                result.changes.append(
                    PromptChange(
                        prompt_name=prompt_config.name,
                        deck_name=task.deck_name,
                        note_key=task.note_key,
                        note_type=task.note_type,
                        field_name=field_name,
                        original_text=original_text,
                        edited_text=edited_text,
                    )
                )

    for deck in selected_copy:
        if id(deck) in changed_decks:
            result.changed_decks.append(deck)

    return result


def run_inline_prompt_on_serialized_collection(
    serialized_data: dict[str, Any],
    include_decks: list[str] | None,
    prompt_config: PromptConfig,
    editor: AsyncInlineBatchEditor,
    *,
    batch_size: int = 1,
    max_in_flight: int = 4,
) -> PromptRunResult:
    """Sync wrapper for async prompt execution."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_inline_prompt_on_serialized_collection_async(
                serialized_data=serialized_data,
                include_decks=include_decks,
                prompt_config=prompt_config,
                editor=editor,
                batch_size=batch_size,
                max_in_flight=max_in_flight,
            )
        )
    raise RuntimeError(
        "Cannot call sync runner inside an active event loop; "
        "use run_inline_prompt_on_serialized_collection_async instead."
    )


def _collect_note_tasks(
    selected_decks: list[dict[str, Any]],
    prompt_config: PromptConfig,
    result: PromptRunResult,
) -> list[_NoteTask]:
    tasks: list[_NoteTask] = []

    for deck in selected_decks:
        deck_name = str(deck.get("name", ""))
        notes = deck.get("notes", [])
        if not isinstance(notes, list):
            continue

        result.processed_decks += 1
        for note in notes:
            if not isinstance(note, dict):
                continue
            result.processed_notes += 1

            note_key = note.get("note_key")
            if not isinstance(note_key, str) or not note_key.strip():
                result.warnings.append(
                    f"{deck_name}/<missing-note_key>: skipped note without note_key"
                )
                continue

            note_type = str(note.get("note_type", ""))
            if not prompt_config.matches_note_type(note_type):
                continue

            fields = note.get("fields", {})
            if not isinstance(fields, dict):
                continue
            string_fields = {
                field_name: value
                for field_name, value in fields.items()
                if isinstance(field_name, str) and isinstance(value, str)
            }
            if not string_fields:
                continue

            target_field_names = _matching_field_names(
                string_fields,
                prompt_config.target_fields,
            )
            if not target_field_names:
                continue

            send_field_names = _matching_field_names(
                string_fields,
                prompt_config.send_fields,
            )
            send_seen = set(send_field_names)
            for field_name in target_field_names:
                if field_name not in send_seen:
                    send_field_names.append(field_name)
                    send_seen.add(field_name)

            note_payload = {
                "note_key": note_key,
                "note_type": note_type,
                "fields": {
                    field_name: string_fields[field_name]
                    for field_name in send_field_names
                },
            }

            result.prompted_notes += 1
            tasks.append(
                _NoteTask(
                    deck_name=deck_name,
                    note_key=note_key,
                    note_type=note_type,
                    note_fields=fields,
                    string_fields=string_fields,
                    target_field_names=target_field_names,
                    send_field_names=send_field_names,
                    note_payload=note_payload,
                    deck_ref=deck,
                )
            )

    return tasks


def _chunk_tasks(tasks: list[_NoteTask], *, batch_size: int) -> list[list[_NoteTask]]:
    return [
        tasks[start_index : start_index + batch_size]
        for start_index in range(0, len(tasks), batch_size)
    ]


async def _dispatch_chunks(
    *,
    prompt_config: PromptConfig,
    editor: AsyncInlineBatchEditor,
    chunks: list[list[_NoteTask]],
    max_in_flight: int,
) -> list[_ChunkResponse]:
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_chunk(index: int, chunk: list[_NoteTask]) -> _ChunkResponse:
        try:
            async with semaphore:
                edited = await editor.edit_batch(
                    prompt_config,
                    [task.note_payload for task in chunk],
                )
            if not isinstance(edited, dict):
                return _ChunkResponse(
                    index=index,
                    chunk=chunk,
                    edited_by_note_key=None,
                    error="response is not a JSON object keyed by note_key",
                )
            return _ChunkResponse(
                index=index,
                chunk=chunk,
                edited_by_note_key=edited,
                error=None,
            )
        except Exception as error:
            return _ChunkResponse(
                index=index,
                chunk=chunk,
                edited_by_note_key=None,
                error=str(error),
            )

    pending = [
        asyncio.create_task(run_chunk(index, chunk))
        for index, chunk in enumerate(chunks)
    ]
    responses = await asyncio.gather(*pending)
    return sorted(responses, key=lambda response: response.index)


def _validate_edited_note(task: _NoteTask, edited: dict[str, Any]) -> str | None:
    if not isinstance(edited, dict):
        return "response is not a JSON object"
    if edited.get("note_key") != task.note_key:
        return "response note_key mismatch"

    edited_fields = edited.get("fields")
    if not isinstance(edited_fields, dict):
        return "response missing fields object"

    missing_or_bad = [
        field_name
        for field_name in task.send_field_names
        if not isinstance(edited_fields.get(field_name), str)
    ]
    if missing_or_bad:
        return f"response fields invalid: {', '.join(missing_or_bad)}"
    return None

"""Prompt runner with async batch execution and deterministic application."""

from __future__ import annotations

import asyncio
import copy
import fnmatch
from dataclasses import dataclass
from typing import Any

from .errors import PromptExecutionError
from .types import (
    AsyncInlineBatchEditor,
    InlineEditedNote,
    InlineNotePayload,
    PromptChange,
    PromptConfig,
    PromptRunOptions,
    PromptRunResult,
)


@dataclass
class _NoteTask:
    deck_name: str
    note_key: str
    note_type: str
    note_fields: dict[str, Any]
    string_fields: dict[str, str]
    target_fields: list[str]
    send_fields: list[str]
    payload: InlineNotePayload
    deck_ref: dict[str, Any]


@dataclass(frozen=True)
class _ChunkResult:
    index: int
    chunk: list[_NoteTask]
    edited_by_note_key: dict[str, InlineEditedNote] | None
    error: str | None


class PromptRunner:
    """Run prompt-driven inline edits over serialized collection JSON."""

    def __init__(self, editor: AsyncInlineBatchEditor):
        self._editor = editor

    async def run_async(
        self,
        serialized_data: dict[str, Any],
        prompt: PromptConfig,
        *,
        options: PromptRunOptions | None = None,
    ) -> PromptRunResult:
        resolved_options = options or PromptRunOptions()
        _validate_options(resolved_options)

        decks = _require_decks(serialized_data)
        selected_decks = _select_decks_with_subdecks(
            decks, resolved_options.include_decks
        )
        working_decks = copy.deepcopy(selected_decks)

        result = PromptRunResult()
        tasks = _collect_tasks(working_decks, prompt, result)
        if not tasks:
            return result

        chunks = _chunk_tasks(tasks, batch_size=resolved_options.batch_size)
        chunk_results = await _dispatch_chunks(
            prompt=prompt,
            editor=self._editor,
            chunks=chunks,
            max_in_flight=resolved_options.max_in_flight,
        )
        changed_deck_ids = _apply_chunk_results(
            chunk_results=chunk_results,
            prompt=prompt,
            result=result,
        )

        for deck in working_decks:
            if id(deck) in changed_deck_ids:
                result.changed_decks.append(deck)

        return result

    def run(
        self,
        serialized_data: dict[str, Any],
        prompt: PromptConfig,
        *,
        options: PromptRunOptions | None = None,
    ) -> PromptRunResult:
        """Sync wrapper for run_async()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_async(
                    serialized_data=serialized_data,
                    prompt=prompt,
                    options=options,
                )
            )
        raise PromptExecutionError(
            "Cannot call sync prompt runner inside an active event loop; "
            "use PromptRunner.run_async instead."
        )


def _select_decks_with_subdecks(
    decks: list[dict[str, Any]],
    include_decks: list[str] | None,
) -> list[dict[str, Any]]:
    """Select decks by recursive include semantics."""
    targets = [name.strip() for name in (include_decks or []) if name.strip()]
    if not targets:
        return decks
    return [
        deck
        for deck in decks
        if any(
            _matches_deck_or_subdeck(str(deck.get("name", "")), target)
            for target in targets
        )
    ]


def _validate_options(options: PromptRunOptions) -> None:
    if options.batch_size <= 0:
        raise PromptExecutionError("batch_size must be > 0")
    if options.max_in_flight <= 0:
        raise PromptExecutionError("max_in_flight must be > 0")


def _require_decks(serialized_data: dict[str, Any]) -> list[dict[str, Any]]:
    decks = serialized_data.get("decks")
    if not isinstance(decks, list):
        raise PromptExecutionError("serialized_data must contain 'decks' list")
    return decks


def _matches_deck_or_subdeck(deck_name: str, target: str) -> bool:
    return deck_name == target or deck_name.startswith(f"{target}::")


def _matching_field_names(fields: dict[str, str], patterns: list[str]) -> list[str]:
    return [
        field_name
        for field_name in fields
        if any(fnmatch.fnmatchcase(field_name, pattern) for pattern in patterns)
    ]


def _collect_tasks(
    selected_decks: list[dict[str, Any]],
    prompt: PromptConfig,
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
            normalized_note_key = note_key.strip()

            note_type = str(note.get("note_type", ""))
            if not prompt.matches_note_type(note_type):
                continue

            fields = note.get("fields")
            if not isinstance(fields, dict):
                continue
            string_fields = {
                field_name: value
                for field_name, value in fields.items()
                if isinstance(field_name, str) and isinstance(value, str)
            }
            if not string_fields:
                continue

            target_fields = _matching_field_names(string_fields, prompt.target_fields)
            if not target_fields:
                continue

            send_fields = _matching_field_names(string_fields, prompt.send_fields)
            send_set = set(send_fields)
            for target_field in target_fields:
                if target_field not in send_set:
                    send_fields.append(target_field)
                    send_set.add(target_field)

            payload = {
                field_name: string_fields[field_name] for field_name in send_fields
            }
            tasks.append(
                _NoteTask(
                    deck_name=deck_name,
                    note_key=normalized_note_key,
                    note_type=note_type,
                    note_fields=fields,
                    string_fields=string_fields,
                    target_fields=target_fields,
                    send_fields=send_fields,
                    payload=InlineNotePayload(
                        note_key=normalized_note_key,
                        note_type=note_type,
                        fields=payload,
                    ),
                    deck_ref=deck,
                )
            )
            result.prompted_notes += 1

    return tasks


def _chunk_tasks(tasks: list[_NoteTask], *, batch_size: int) -> list[list[_NoteTask]]:
    return [
        tasks[index : index + batch_size] for index in range(0, len(tasks), batch_size)
    ]


async def _dispatch_chunks(
    *,
    prompt: PromptConfig,
    editor: AsyncInlineBatchEditor,
    chunks: list[list[_NoteTask]],
    max_in_flight: int,
) -> list[_ChunkResult]:
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_chunk(index: int, chunk: list[_NoteTask]) -> _ChunkResult:
        try:
            async with semaphore:
                edited = await editor.edit_notes(
                    prompt,
                    [task.payload for task in chunk],
                )
            if not isinstance(edited, dict):
                return _ChunkResult(
                    index=index,
                    chunk=chunk,
                    edited_by_note_key=None,
                    error="response is not a JSON object keyed by note_key",
                )
            return _ChunkResult(
                index=index,
                chunk=chunk,
                edited_by_note_key=edited,
                error=None,
            )
        except Exception as error:
            return _ChunkResult(
                index=index,
                chunk=chunk,
                edited_by_note_key=None,
                error=str(error),
            )

    pending = [
        asyncio.create_task(run_chunk(index, chunk))
        for index, chunk in enumerate(chunks)
    ]
    chunk_results = await asyncio.gather(*pending)
    return sorted(chunk_results, key=lambda item: item.index)


def _apply_chunk_results(
    *,
    chunk_results: list[_ChunkResult],
    prompt: PromptConfig,
    result: PromptRunResult,
) -> set[int]:
    changed_deck_ids: set[int] = set()

    for chunk_result in chunk_results:
        if chunk_result.error is not None:
            for task in chunk_result.chunk:
                result.warnings.append(
                    f"{task.deck_name}/{task.note_key}: {chunk_result.error}"
                )
            continue

        edited_by_note_key = chunk_result.edited_by_note_key or {}
        for task in chunk_result.chunk:
            edited_note = edited_by_note_key.get(task.note_key)
            if edited_note is None:
                result.warnings.append(
                    f"{task.deck_name}/{task.note_key}: response missing note_key"
                )
                continue

            error = _validate_edited_note(task, edited_note)
            if error:
                result.warnings.append(f"{task.deck_name}/{task.note_key}: {error}")
                continue

            if _apply_note_changes(task, edited_note, prompt.name, result):
                changed_deck_ids.add(id(task.deck_ref))

    return changed_deck_ids


def _validate_edited_note(
    task: _NoteTask,
    edited_note: InlineEditedNote,
) -> str | None:
    if not isinstance(edited_note, InlineEditedNote):
        return "response is not a valid edited note"
    if edited_note.note_key != task.note_key:
        return "response note_key mismatch"

    invalid_fields = [
        field_name
        for field_name in task.send_fields
        if not isinstance(edited_note.fields.get(field_name), str)
    ]
    if invalid_fields:
        return f"response fields invalid: {', '.join(invalid_fields)}"
    return None


def _apply_note_changes(
    task: _NoteTask,
    edited_note: InlineEditedNote,
    prompt_name: str,
    result: PromptRunResult,
) -> bool:
    edited_fields = edited_note.fields
    changed_any = False

    for field_name in task.target_fields:
        original_text = task.string_fields[field_name]
        edited_text = edited_fields[field_name]
        if edited_text == original_text:
            continue

        task.note_fields[field_name] = edited_text
        result.changed_fields += 1
        result.changes.append(
            PromptChange(
                prompt_name=prompt_name,
                deck_name=task.deck_name,
                note_key=task.note_key,
                note_type=task.note_type,
                field_name=field_name,
                original_text=original_text,
                edited_text=edited_text,
            )
        )
        changed_any = True

    return changed_any

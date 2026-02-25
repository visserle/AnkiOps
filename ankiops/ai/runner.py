"""Prompt runner with streaming async batch execution and deterministic apply."""

from __future__ import annotations

import asyncio
import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class _GlobMatcher:
    patterns: tuple[str, ...]

    @classmethod
    def from_patterns(cls, patterns: list[str]) -> _GlobMatcher:
        return cls(patterns=tuple(patterns))

    def matches(self, value: str) -> bool:
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in self.patterns)

    def select_names(self, names: Iterable[str]) -> list[str]:
        return [name for name in names if self.matches(name)]


@dataclass(frozen=True)
class _PromptMatchers:
    note_types: _GlobMatcher
    target_fields: _GlobMatcher
    send_fields: _GlobMatcher

    @classmethod
    def from_prompt(cls, prompt: PromptConfig) -> _PromptMatchers:
        return cls(
            note_types=_GlobMatcher.from_patterns(prompt.note_types),
            target_fields=_GlobMatcher.from_patterns(prompt.target_fields),
            send_fields=_GlobMatcher.from_patterns(prompt.send_fields),
        )


@dataclass
class _NoteTask:
    deck_name: str
    note_key: str
    note_type: str
    note_fields: dict[str, Any]
    original_target_fields: dict[str, str]
    target_fields: tuple[str, ...]
    payload: InlineNotePayload
    deck_index: int


@dataclass(frozen=True)
class _ChunkResult:
    chunk: list[_NoteTask]
    edited_by_note_key: dict[str, InlineEditedNote] | None
    error: str | None


@runtime_checkable
class _AsyncCloseable(Protocol):
    async def aclose(self) -> None:
        """Close any asynchronous resources held by an editor."""


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
            decks,
            resolved_options.include_decks,
        )
        result = PromptRunResult()
        matchers = _PromptMatchers.from_prompt(prompt)

        changed_deck_indexes: set[int] = set()
        try:
            changed_deck_indexes = await _process_task_stream(
                selected_decks=selected_decks,
                prompt=prompt,
                editor=self._editor,
                matchers=matchers,
                batch_size=resolved_options.batch_size,
                max_in_flight=resolved_options.max_in_flight,
                max_warnings=resolved_options.max_warnings,
                result=result,
            )
        finally:
            await _close_editor_if_supported(self._editor)

        for deck_index, deck in enumerate(selected_decks):
            if deck_index in changed_deck_indexes:
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


async def _process_task_stream(
    *,
    selected_decks: list[dict[str, Any]],
    prompt: PromptConfig,
    editor: AsyncInlineBatchEditor,
    matchers: _PromptMatchers,
    batch_size: int,
    max_in_flight: int,
    max_warnings: int,
    result: PromptRunResult,
) -> set[int]:
    tasks = _iter_note_tasks(
        selected_decks=selected_decks,
        matchers=matchers,
        max_warnings=max_warnings,
        result=result,
    )
    chunks = _iter_task_chunks(tasks, batch_size=batch_size)
    return await _dispatch_chunks_and_apply(
        chunks=chunks,
        prompt=prompt,
        editor=editor,
        max_in_flight=max_in_flight,
        max_warnings=max_warnings,
        result=result,
    )


def _iter_note_tasks(
    *,
    selected_decks: list[dict[str, Any]],
    matchers: _PromptMatchers,
    max_warnings: int,
    result: PromptRunResult,
) -> Iterator[_NoteTask]:
    for deck_index, deck in enumerate(selected_decks):
        deck_name = str(deck.get("name", ""))
        notes = deck.get("notes")
        if not isinstance(notes, list):
            continue

        result.processed_decks += 1
        for note in notes:
            if not isinstance(note, dict):
                continue
            result.processed_notes += 1

            note_key = _normalize_note_key(note.get("note_key"))
            if note_key is None:
                _add_warning(
                    result,
                    f"{deck_name}/<missing-note_key>: skipped note without note_key",
                    max_warnings=max_warnings,
                )
                continue

            note_type = _normalize_note_type(note.get("note_type"))
            if note_type is None:
                _add_warning(
                    result,
                    f"{deck_name}/{note_key}: skipped note without note_type",
                    max_warnings=max_warnings,
                )
                continue
            if not matchers.note_types.matches(note_type):
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

            target_fields = matchers.target_fields.select_names(string_fields.keys())
            if not target_fields:
                continue
            send_fields = _merge_send_and_target_fields(
                send_fields=matchers.send_fields.select_names(string_fields.keys()),
                target_fields=target_fields,
            )

            payload_fields = {
                field_name: string_fields[field_name] for field_name in send_fields
            }
            target_snapshot = {
                field_name: string_fields[field_name] for field_name in target_fields
            }
            result.prompted_notes += 1
            yield _NoteTask(
                deck_name=deck_name,
                note_key=note_key,
                note_type=note_type,
                note_fields=fields,
                original_target_fields=target_snapshot,
                target_fields=tuple(target_fields),
                payload=InlineNotePayload(
                    note_key=note_key,
                    note_type=note_type,
                    fields=payload_fields,
                ),
                deck_index=deck_index,
            )


def _normalize_note_key(raw_note_key: Any) -> str | None:
    if not isinstance(raw_note_key, str):
        return None
    note_key = raw_note_key.strip()
    return note_key or None


def _normalize_note_type(raw_note_type: Any) -> str | None:
    if not isinstance(raw_note_type, str):
        return None
    note_type = raw_note_type.strip()
    return note_type or None


def _merge_send_and_target_fields(
    *,
    send_fields: list[str],
    target_fields: list[str],
) -> list[str]:
    merged = list(send_fields)
    seen = set(merged)
    for field_name in target_fields:
        if field_name not in seen:
            merged.append(field_name)
            seen.add(field_name)
    return merged


def _iter_task_chunks(
    tasks: Iterator[_NoteTask],
    *,
    batch_size: int,
) -> Iterator[list[_NoteTask]]:
    chunk: list[_NoteTask] = []
    for task in tasks:
        chunk.append(task)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


async def _dispatch_chunks_and_apply(
    *,
    chunks: Iterator[list[_NoteTask]],
    prompt: PromptConfig,
    editor: AsyncInlineBatchEditor,
    max_in_flight: int,
    max_warnings: int,
    result: PromptRunResult,
) -> set[int]:
    changed_deck_indexes: set[int] = set()
    pending: set[asyncio.Task[_ChunkResult]] = set()

    for chunk in chunks:
        pending.add(asyncio.create_task(_run_chunk(prompt, editor, chunk)))
        if len(pending) >= max_in_flight:
            pending = await _drain_first_completed(
                pending=pending,
                prompt_name=prompt.name,
                changed_deck_indexes=changed_deck_indexes,
                max_warnings=max_warnings,
                result=result,
            )

    while pending:
        pending = await _drain_first_completed(
            pending=pending,
            prompt_name=prompt.name,
            changed_deck_indexes=changed_deck_indexes,
            max_warnings=max_warnings,
            result=result,
        )

    return changed_deck_indexes


async def _run_chunk(
    prompt: PromptConfig,
    editor: AsyncInlineBatchEditor,
    chunk: list[_NoteTask],
) -> _ChunkResult:
    try:
        edited = await editor.edit_notes(
            prompt,
            [task.payload for task in chunk],
        )
        if not isinstance(edited, dict):
            return _ChunkResult(
                chunk=chunk,
                edited_by_note_key=None,
                error="response is not a JSON object keyed by note_key",
            )
        return _ChunkResult(
            chunk=chunk,
            edited_by_note_key=edited,
            error=None,
        )
    except Exception as error:
        return _ChunkResult(
            chunk=chunk,
            edited_by_note_key=None,
            error=str(error),
        )


async def _drain_first_completed(
    *,
    pending: set[asyncio.Task[_ChunkResult]],
    prompt_name: str,
    changed_deck_indexes: set[int],
    max_warnings: int,
    result: PromptRunResult,
) -> set[asyncio.Task[_ChunkResult]]:
    done, remaining = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for completed in done:
        _apply_chunk_result(
            chunk_result=completed.result(),
            prompt_name=prompt_name,
            changed_deck_indexes=changed_deck_indexes,
            max_warnings=max_warnings,
            result=result,
        )
    return set(remaining)


def _apply_chunk_result(
    *,
    chunk_result: _ChunkResult,
    prompt_name: str,
    changed_deck_indexes: set[int],
    max_warnings: int,
    result: PromptRunResult,
) -> None:
    if chunk_result.error is not None:
        for task in chunk_result.chunk:
            _add_warning(
                result,
                f"{task.deck_name}/{task.note_key}: {chunk_result.error}",
                max_warnings=max_warnings,
            )
        return

    edited_by_note_key = chunk_result.edited_by_note_key or {}
    for task in chunk_result.chunk:
        edited_note = edited_by_note_key.get(task.note_key)
        if edited_note is None:
            _add_warning(
                result,
                f"{task.deck_name}/{task.note_key}: response missing note_key",
                max_warnings=max_warnings,
            )
            continue

        error = _validate_edited_note(task, edited_note)
        if error:
            _add_warning(
                result,
                f"{task.deck_name}/{task.note_key}: {error}",
                max_warnings=max_warnings,
            )
            continue

        if _apply_note_changes(task, edited_note, prompt_name, result):
            changed_deck_indexes.add(task.deck_index)


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
        for field_name in task.target_fields
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
    changed_any = False
    for field_name in task.target_fields:
        original_text = task.original_target_fields[field_name]
        edited_text = edited_note.fields[field_name]
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


def _select_decks_with_subdecks(
    decks: list[dict[str, Any]],
    include_decks: list[str] | None,
) -> list[dict[str, Any]]:
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
    if options.max_warnings <= 0:
        raise PromptExecutionError("max_warnings must be > 0")


def _require_decks(serialized_data: dict[str, Any]) -> list[dict[str, Any]]:
    decks = serialized_data.get("decks")
    if not isinstance(decks, list):
        raise PromptExecutionError("serialized_data must contain 'decks' list")
    return decks


def _matches_deck_or_subdeck(deck_name: str, target: str) -> bool:
    return deck_name == target or deck_name.startswith(f"{target}::")


def _add_warning(
    result: PromptRunResult,
    warning: str,
    *,
    max_warnings: int,
) -> None:
    if len(result.warnings) < max_warnings:
        result.warnings.append(warning)
        return
    result.dropped_warnings += 1


async def _close_editor_if_supported(editor: AsyncInlineBatchEditor) -> None:
    if isinstance(editor, _AsyncCloseable):
        await editor.aclose()

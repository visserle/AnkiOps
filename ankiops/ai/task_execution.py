"""Task batch execution and chunk application helpers."""

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

from ankiops.ai.task_apply import add_warning, apply_note_changes, validate_edited_note
from ankiops.ai.task_selection import NoteTask, TaskMatchers, iter_note_tasks
from ankiops.ai.types import (
    AsyncInlineBatchEditor,
    InlineEditedNote,
    TaskConfig,
    TaskProgressUpdate,
    TaskRunResult,
)

_CLOZE_REQUIREMENT = (
    "For cloze note types, preserve valid cloze markers "
    "(for example {{c1::answer}}). Never remove or break cloze syntax."
)


@dataclass(frozen=True)
class ChunkResult:
    chunk: list[NoteTask]
    edited_by_note_key: dict[str, InlineEditedNote] | None
    error: str | None


async def process_task_stream(
    *,
    selected_decks: list[dict[str, Any]],
    task_config: TaskConfig,
    editor: AsyncInlineBatchEditor,
    matchers: TaskMatchers,
    batch_size: int,
    max_in_flight: int,
    max_warnings: int,
    progress_callback: Callable[[TaskProgressUpdate], None] | None,
    result: TaskRunResult,
) -> set[int]:
    """Run all matched notes through async chunk dispatch and apply results."""
    note_tasks = iter_note_tasks(
        selected_decks=selected_decks,
        matchers=matchers,
        max_warnings=max_warnings,
        result=result,
    )
    chunks = iter_task_chunks(note_tasks, batch_size=batch_size)
    return await dispatch_chunks_and_apply(
        chunks=chunks,
        task_config=task_config,
        editor=editor,
        max_in_flight=max_in_flight,
        max_warnings=max_warnings,
        progress_callback=progress_callback,
        result=result,
    )


def iter_task_chunks(
    note_tasks: Iterator[NoteTask],
    *,
    batch_size: int,
) -> Iterator[list[NoteTask]]:
    """Chunk note tasks into configured batch sizes."""
    chunk: list[NoteTask] = []
    for note_task in note_tasks:
        chunk.append(note_task)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


async def dispatch_chunks_and_apply(
    *,
    chunks: Iterator[list[NoteTask]],
    task_config: TaskConfig,
    editor: AsyncInlineBatchEditor,
    max_in_flight: int,
    max_warnings: int,
    progress_callback: Callable[[TaskProgressUpdate], None] | None,
    result: TaskRunResult,
) -> set[int]:
    """Dispatch chunks concurrently and apply results as they complete."""
    changed_deck_indexes: set[int] = set()
    pending: set[asyncio.Task[ChunkResult]] = set()
    queued_chunks = 0
    completed_chunks = 0
    started_at = time.perf_counter()

    for chunk in chunks:
        pending.add(asyncio.create_task(run_chunk(task_config, editor, chunk)))
        queued_chunks += 1
        if len(pending) >= max_in_flight:
            pending, completed_count = await drain_first_completed(
                pending=pending,
                task_id=task_config.id,
                changed_deck_indexes=changed_deck_indexes,
                max_warnings=max_warnings,
                result=result,
            )
            completed_chunks += completed_count
            emit_progress_update(
                progress_callback=progress_callback,
                started_at=started_at,
                queued_chunks=queued_chunks,
                completed_chunks=completed_chunks,
                in_flight=len(pending),
                result=result,
            )

    while pending:
        pending, completed_count = await drain_first_completed(
            pending=pending,
            task_id=task_config.id,
            changed_deck_indexes=changed_deck_indexes,
            max_warnings=max_warnings,
            result=result,
        )
        completed_chunks += completed_count
        emit_progress_update(
            progress_callback=progress_callback,
            started_at=started_at,
            queued_chunks=queued_chunks,
            completed_chunks=completed_chunks,
            in_flight=len(pending),
            result=result,
        )

    return changed_deck_indexes


async def run_chunk(
    task_config: TaskConfig,
    editor: AsyncInlineBatchEditor,
    chunk: list[NoteTask],
) -> ChunkResult:
    """Run one chunk through editor and normalize top-level shape checks."""
    effective_task_config = _task_config_for_chunk(task_config, chunk)
    try:
        edited = await editor.edit_notes(
            effective_task_config,
            [task_item.payload for task_item in chunk],
        )
        if not isinstance(edited, dict):
            return ChunkResult(
                chunk=chunk,
                edited_by_note_key=None,
                error="response is not a JSON object keyed by note_key",
            )
        return ChunkResult(
            chunk=chunk,
            edited_by_note_key=edited,
            error=None,
        )
    except Exception as error:
        return ChunkResult(
            chunk=chunk,
            edited_by_note_key=None,
            error=_format_chunk_error(error),
        )


async def drain_first_completed(
    *,
    pending: set[asyncio.Task[ChunkResult]],
    task_id: str,
    changed_deck_indexes: set[int],
    max_warnings: int,
    result: TaskRunResult,
) -> tuple[set[asyncio.Task[ChunkResult]], int]:
    """Wait for first completed chunk task and apply its changes."""
    done, remaining = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for completed in done:
        apply_chunk_result(
            chunk_result=completed.result(),
            task_id=task_id,
            changed_deck_indexes=changed_deck_indexes,
            max_warnings=max_warnings,
            result=result,
        )
    return set(remaining), len(done)


def apply_chunk_result(
    *,
    chunk_result: ChunkResult,
    task_id: str,
    changed_deck_indexes: set[int],
    max_warnings: int,
    result: TaskRunResult,
) -> None:
    """Apply one completed chunk result to in-memory notes."""
    if chunk_result.error is not None:
        for note_task in chunk_result.chunk:
            add_warning(
                result,
                f"{note_task.deck_name}/{note_task.note_key}: {chunk_result.error}",
                max_warnings=max_warnings,
            )
        return

    edited_by_note_key = chunk_result.edited_by_note_key or {}
    for note_task in chunk_result.chunk:
        edited_note = edited_by_note_key.get(note_task.note_key)
        if edited_note is None:
            add_warning(
                result,
                (
                    f"{note_task.deck_name}/{note_task.note_key}: "
                    "response missing note_key"
                ),
                max_warnings=max_warnings,
            )
            continue

        error = validate_edited_note(note_task, edited_note)
        if error:
            add_warning(
                result,
                f"{note_task.deck_name}/{note_task.note_key}: {error}",
                max_warnings=max_warnings,
            )
            continue

        if apply_note_changes(note_task, edited_note, task_id, result):
            changed_deck_indexes.add(note_task.deck_index)


def _format_chunk_error(error: Exception) -> str:
    error_message = str(error).strip()
    if not error_message:
        return error.__class__.__name__
    return f"{error.__class__.__name__}: {error_message}"


def _task_config_for_chunk(
    task_config: TaskConfig,
    chunk: list[NoteTask],
) -> TaskConfig:
    if not _chunk_has_cloze_note_type(chunk):
        return task_config

    normalized_instructions = task_config.instructions.rstrip()
    if _CLOZE_REQUIREMENT in normalized_instructions:
        return task_config
    return replace(
        task_config,
        instructions=f"{normalized_instructions}\n{_CLOZE_REQUIREMENT}",
    )


def _chunk_has_cloze_note_type(chunk: list[NoteTask]) -> bool:
    return any("cloze" in note_task.note_type.casefold() for note_task in chunk)


def emit_progress_update(
    *,
    progress_callback: Callable[[TaskProgressUpdate], None] | None,
    started_at: float,
    queued_chunks: int,
    completed_chunks: int,
    in_flight: int,
    result: TaskRunResult,
) -> None:
    if progress_callback is None:
        return
    warning_count = len(result.warnings) + result.dropped_warnings
    progress_callback(
        TaskProgressUpdate(
            elapsed_seconds=max(0.0, time.perf_counter() - started_at),
            queued_chunks=queued_chunks,
            completed_chunks=completed_chunks,
            in_flight=in_flight,
            processed_decks=result.processed_decks,
            processed_notes=result.processed_notes,
            matched_notes=result.matched_notes,
            changed_fields=result.changed_fields,
            warning_count=warning_count,
        )
    )

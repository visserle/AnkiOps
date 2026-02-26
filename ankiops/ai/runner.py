"""Task runner orchestration facade."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from ankiops.ai.errors import TaskExecutionError
from ankiops.ai.task_execution import process_task_stream
from ankiops.ai.task_selection import TaskMatchers, require_decks, select_decks
from ankiops.ai.types import (
    AsyncInlineBatchEditor,
    TaskConfig,
    TaskRunOptions,
    TaskRunResult,
)


@runtime_checkable
class _AsyncCloseable(Protocol):
    async def aclose(self) -> None:
        """Close any asynchronous resources held by an editor."""


class TaskRunner:
    """Run task-driven inline edits over serialized collection JSON."""

    def __init__(self, editor: AsyncInlineBatchEditor):
        self._editor = editor

    async def run_async(
        self,
        serialized_data: dict[str, Any],
        task_config: TaskConfig,
        *,
        options: TaskRunOptions | None = None,
    ) -> TaskRunResult:
        resolved_options = _resolve_options(task_config, options)
        _validate_options(resolved_options)

        decks = require_decks(serialized_data)
        task_scoped_decks = select_decks(
            decks,
            include_decks=task_config.scope_decks,
            include_subdecks=task_config.scope_subdecks,
        )
        selected_decks = select_decks(
            task_scoped_decks,
            include_decks=resolved_options.include_decks,
            include_subdecks=True,
        )
        result = TaskRunResult()
        matchers = TaskMatchers.from_task(task_config)

        try:
            changed_deck_indexes = await process_task_stream(
                selected_decks=selected_decks,
                task_config=task_config,
                editor=self._editor,
                matchers=matchers,
                batch_size=resolved_options.batch_size or 1,
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
        task_config: TaskConfig,
        *,
        options: TaskRunOptions | None = None,
    ) -> TaskRunResult:
        """Sync wrapper for run_async()."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_async(
                    serialized_data=serialized_data,
                    task_config=task_config,
                    options=options,
                )
            )
        raise TaskExecutionError(
            "Cannot call sync task runner inside an active event loop; "
            "use TaskRunner.run_async instead."
        )


def _resolve_options(
    task_config: TaskConfig,
    options: TaskRunOptions | None,
) -> TaskRunOptions:
    if options is None:
        return TaskRunOptions(batch_size=task_config.batch_size)
    return TaskRunOptions(
        include_decks=options.include_decks,
        batch_size=(
            task_config.batch_size if options.batch_size is None else options.batch_size
        ),
        max_in_flight=options.max_in_flight,
        max_warnings=options.max_warnings,
    )


def _validate_options(options: TaskRunOptions) -> None:
    if options.batch_size is None or options.batch_size <= 0:
        raise TaskExecutionError("batch_size must be > 0")
    if options.max_in_flight <= 0:
        raise TaskExecutionError("max_in_flight must be > 0")
    if options.max_warnings <= 0:
        raise TaskExecutionError("max_warnings must be > 0")


async def _close_editor_if_supported(editor: AsyncInlineBatchEditor) -> None:
    if isinstance(editor, _AsyncCloseable):
        await editor.aclose()

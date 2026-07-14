"""Command-line entrypoints for the LLM pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich import get_console as rich_get_console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ankiops.collection import (
    LLM_DIR,
    file_stem_to_deck_name,
    require_collection_root,
)
from ankiops.deck_sources import load_note_types_for_collection

from .execution import TaskExecutionProgress, run_task
from .jobs import LlmJobRequestNoteRef, show_job
from .jobs import list_jobs as list_llm_jobs
from .models import MODEL_REGISTRY_FILE_NAME
from .planning import plan_task
from .tasks import load_llm_task_catalog

logger = logging.getLogger(__name__)


def configure_llm_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handler: Callable[[argparse.Namespace], None],
) -> None:
    llm_parser = subparsers.add_parser(
        "llm",
        help="Plan/run LLM jobs and inspect LLM job history",
    )
    llm_parser.add_argument("task_name", nargs="?", help="Task name to plan/run")
    llm_parser.add_argument(
        "--run",
        action="store_true",
        help="Run task job (default with task is dry-run plan)",
    )
    llm_parser.add_argument(
        "--job",
        dest="job_id",
        help="Select one LLM job (numeric id or 'latest')",
    )
    llm_parser.add_argument(
        "--model",
        help=(
            "Override model for this plan/run "
            f"(must exist in {LLM_DIR}/{MODEL_REGISTRY_FILE_NAME})"
        ),
    )
    llm_parser.add_argument(
        "--deck",
        help=(
            "Override task scope to one exact deck (includes subdecks by default). "
            "Accepts 'Parent::Child' or markdown alias 'Parent__Child[.md]'."
        ),
    )
    llm_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip automatic git snapshot (with <task> --run)",
    )
    llm_parser.set_defaults(handler=handler)


def run_llm(
    args: argparse.Namespace,
) -> None:
    """Status/plan/run/show for LLM jobs."""
    collection_root = require_collection_root()
    task_name = getattr(args, "task_name", None)
    run_mode = bool(getattr(args, "run", False))
    job_id = getattr(args, "job_id", None)
    model_override = getattr(args, "model", None)
    deck_override = _normalize_deck_override(getattr(args, "deck", None))
    no_auto_commit = bool(getattr(args, "no_auto_commit", False))

    if job_id is not None:
        _show_job(
            task_name=task_name,
            run_mode=run_mode,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            collection_root=collection_root,
            job_id=job_id,
        )
        return

    if task_name is None:
        _show_status(
            run_mode=run_mode,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            collection_root=collection_root,
        )
        return

    if no_auto_commit and not run_mode:
        _usage_error("--no-auto-commit requires --run.")

    if run_mode:
        _run_task(
            collection_root=collection_root,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
        )
        return

    _show_plan(
        collection_root=collection_root,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
    )


def _show_job(
    *,
    task_name: str | None,
    run_mode: bool,
    model_override: str | None,
    deck_override: str | None,
    no_auto_commit: bool,
    collection_root: Path,
    job_id: str,
) -> None:
    if task_name is not None:
        _usage_error("Cannot combine <task> with --job.")
    if run_mode:
        _usage_error("Cannot combine --run with --job.")
    if model_override is not None:
        _usage_error("--model requires <task>.")
    if deck_override is not None:
        _usage_error("--deck requires <task>.")
    if no_auto_commit:
        _usage_error("--no-auto-commit requires <task> --run.")

    try:
        detail = show_job(collection_root=collection_root, job_id=job_id)
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error
    if detail is None:
        logger.error(f"Unknown LLM job '{job_id}'")
        raise SystemExit(1)

    logger.info(
        "Job %s - %s (%s / %s)",
        detail.job_id,
        detail.task_name,
        detail.model,
        detail.model_id,
    )
    logger.info(
        "Status: %s (persisted=%s)",
        detail.status.value,
        "yes" if detail.persisted else "no",
    )
    logger.info(
        "Timing: created=%s started=%s finished=%s",
        detail.created_at,
        detail.started_at,
        detail.finished_at or "-",
    )
    logger.info("Summary: %s", detail.summary.format())
    logger.info("Usage: %s", detail.summary.format_usage())
    logger.info("Cost: %s", detail.summary.format_cost())
    if detail.fatal_error:
        logger.error("Fatal error: %s", detail.fatal_error)
    logger.info("")
    logger.info("Items:")
    if not detail.items:
        logger.info("  none")
        return

    item_rows: list[list[str]] = []
    for item in detail.items:
        item_rows.append(
            [
                str(item.ordinal),
                item.note_key or "unknown",
                item.source,
                item.deck_name,
                item.note_type or "unknown",
                item.item_status.value,
                str(item.request_count),
                _format_field_list(item.changed_fields),
            ]
        )

    _log_table(
        [
            "#",
            "Note",
            "Source",
            "Deck",
            "Type",
            "Final",
            "Requests",
            "Changed",
        ],
        item_rows,
    )
    for item in detail.items:
        if item.error_message:
            logger.error("  #%d error=%s", item.ordinal, item.error_message)

    logger.info("")
    logger.info("Requests:")
    if not detail.requests:
        logger.info("  none")
        return

    request_rows: list[list[str]] = []
    for request in detail.requests:
        request_rows.append(
            [
                str(request.request_id),
                request.outcome,
                f"{_format_count(request.input_tokens)}/{_format_count(request.output_tokens)}",
                f"{request.latency_ms / 1000:.2f}s",
                _format_request_notes(request.notes),
                request.error_message or "-",
            ]
        )

    _log_table(
        ["Request", "Outcome", "Tokens", "Latency", "Notes", "Error"],
        request_rows,
    )


def _show_status(
    *,
    run_mode: bool,
    model_override: str | None,
    deck_override: str | None,
    no_auto_commit: bool,
    collection_root: Path,
) -> None:
    if run_mode:
        _usage_error("--run requires <task>.")
    if model_override is not None:
        _usage_error("--model requires <task>.")
    if deck_override is not None:
        _usage_error("--deck requires <task>.")
    if no_auto_commit:
        _usage_error("--no-auto-commit requires <task> --run.")

    note_type_configs = load_note_types_for_collection(collection_root)
    catalog = load_llm_task_catalog(
        collection_root,
        note_type_configs=note_type_configs,
    )
    logger.info("LLM config: %s", "OK" if not catalog.errors else "INVALID")
    logger.info("")
    logger.info("Tasks:")
    tasks = sorted(catalog.tasks_by_name.values(), key=lambda task: task.name)
    if not tasks:
        logger.info("  none")
    else:
        _log_table(
            ["Name", "Model", "Field rules"],
            [
                [task.name, str(task.model), str(len(task.field_rules))]
                for task in tasks
            ],
        )

    logger.info("")
    logger.info("Recent jobs:")
    jobs = list_llm_jobs(collection_root=collection_root)
    if not jobs:
        logger.info("  none")
    else:
        _log_table(
            ["Job", "Task", "Model", "Status", "Persisted", "Created"],
            [
                [
                    str(job.job_id),
                    job.task_name,
                    job.model,
                    job.status.value,
                    "yes" if job.persisted else "no",
                    job.created_at,
                ]
                for job in jobs
            ],
        )

    if catalog.errors:
        for message in catalog.errors.values():
            logger.error(message)
        raise SystemExit(1)


def _run_task(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None,
    deck_override: str | None,
    no_auto_commit: bool,
) -> None:
    try:
        with _llm_progress_callback() as progress_callback:
            result = run_task(
                collection_root=collection_root,
                task_name=task_name,
                model_override=model_override,
                deck_override=deck_override,
                no_auto_commit=no_auto_commit,
                progress_callback=progress_callback,
            )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    inspect_command = f"ankiops llm --job {result.job_id}"
    if result.failed:
        logger.error(
            "LLM job #%d failed with %d error(s). Cost: %s. Inspect: %s",
            result.job_id,
            result.summary.errors,
            result.summary.format_cost(),
            inspect_command,
        )
        raise SystemExit(1)
    logger.info(
        "LLM job #%d completed%s. Cost: %s. Inspect: %s",
        result.job_id,
        " (no markdown changes persisted)" if not result.persisted else "",
        result.summary.format_cost(),
        inspect_command,
    )


def _show_plan(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None,
    deck_override: str | None,
) -> None:
    try:
        plan = plan_task(
            collection_root=collection_root,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
        )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    logger.info("Plan: %s (model=%s)", plan.task_name, plan.model)
    logger.info("Deck scope: %s", plan.deck_scope)
    logger.info("Serializer scope: %s", plan.serializer_scope)
    if plan.system_prompt_path is not None:
        logger.info("System prompt file: %s", plan.system_prompt_path)
    if plan.user_prompt_path is not None:
        logger.info("User prompt file: %s", plan.user_prompt_path)
    logger.info("Request defaults: %s", plan.request_defaults)
    logger.info(
        "Discovery: decks_seen=%d decks_matched=%d notes_seen=%d "
        "eligible=%d skipped=%d errors=%d",
        plan.summary.decks_seen,
        plan.summary.decks_matched,
        plan.summary.notes_seen,
        plan.summary.eligible,
        plan.summary.skipped,
        plan.summary.errors,
    )
    logger.info("")
    logger.info("Field surface:")
    if plan.field_surface:
        _log_table(
            ["Source", "Type", "Candidates", "Tags", "Editable", "Read-only", "Hidden"],
            [
                [
                    surface.source,
                    surface.note_type,
                    str(surface.candidate_notes),
                    surface.tag_access.value,
                    _format_field_list(surface.editable_fields),
                    _format_field_list(surface.read_only_fields),
                    _format_field_list(surface.hidden_fields),
                ]
                for surface in plan.field_surface
            ],
        )
    else:
        logger.info("  none")
    logger.info("")
    logger.info("Full prompt:")
    logger.info("%s", plan.format_full_prompt())
    logger.info("")
    logger.info("Request estimate: %s", _format_count(plan.requests_estimate))
    logger.info(
        "Cost estimate (assuming number of input tokens equals output tokens): %s",
        plan.format_cost_estimate(),
    )
    run_command = f"ankiops llm {plan.task_name} --run"
    if model_override:
        run_command = f"{run_command} --model {model_override}"
    if deck_override is not None:
        run_command = f"{run_command} --deck {shlex.quote(deck_override)}"
    logger.info("To run this task: %s", run_command)


def _normalize_deck_override(value: str | None) -> str | None:
    if value is None:
        return None
    deck_input = value.strip()
    if not deck_input:
        _usage_error("--deck requires a non-empty deck name.")

    trimmed_md_suffix = False
    if deck_input.lower().endswith(".md"):
        deck_input = deck_input[:-3]
        trimmed_md_suffix = True
        if not deck_input:
            _usage_error("--deck requires a non-empty deck name.")

    if trimmed_md_suffix or "__" in deck_input:
        deck_name = file_stem_to_deck_name(deck_input)
    else:
        deck_name = deck_input

    if any(char in deck_name for char in ("*", "?", "[")):
        _usage_error("--deck must be an exact deck name (wildcards are not supported).")
    return deck_name


def _usage_error(message: str) -> None:
    logger.error(message)
    raise SystemExit(2)


def _format_field_list(fields: list[str]) -> str:
    return ", ".join(fields) if fields else "-"


def _format_request_notes(notes: Iterable[LlmJobRequestNoteRef]) -> str:
    formatted = [f"#{note.ordinal} {note.note_key or 'unknown'}" for note in notes]
    return ", ".join(formatted) if formatted else "-"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_width: int | None = None,
) -> list[str]:
    if not headers:
        return []
    if max_width is None:
        max_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    max_width = max(max_width, 40)

    separator_width = 2 * (len(headers) - 1)
    max_cell_lengths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            max_cell_lengths[index] = max(max_cell_lengths[index], len(cell))

    widths = [len(header) for header in headers]
    available = max(0, max_width - separator_width - sum(widths))
    expansion_needs = [
        max_cell_lengths[index] - widths[index] for index in range(len(headers))
    ]
    while available > 0:
        grew = False
        for index, need in enumerate(expansion_needs):
            if available == 0:
                break
            if need <= 0:
                continue
            widths[index] += 1
            expansion_needs[index] -= 1
            available -= 1
            grew = True
        if not grew:
            break

    def wrap_cell(value: str, width: int) -> list[str]:
        wrapped = textwrap.wrap(
            value,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if wrapped:
            return wrapped
        if not value:
            return [""]
        return textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=True,
        )

    def render_row(values: list[str]) -> list[str]:
        wrapped_cells = [
            wrap_cell(values[index], widths[index]) for index in range(len(values))
        ]
        row_height = max(len(lines) for lines in wrapped_cells)
        rendered: list[str] = []
        for line_index in range(row_height):
            rendered.append(
                "  ".join(
                    (
                        wrapped_cells[column_index][line_index]
                        if line_index < len(wrapped_cells[column_index])
                        else ""
                    ).ljust(widths[column_index])
                    for column_index in range(len(values))
                ).rstrip()
            )
        return rendered

    rendered_lines = [
        *render_row(headers),
        *render_row(["-" * width for width in widths]),
    ]
    for row in rows:
        rendered_lines.extend(render_row(row))
    return rendered_lines


def _log_table(headers: list[str], rows: list[list[str]]) -> None:
    for line in _format_table(headers, rows):
        logger.info(line)


def _is_progress_render_enabled() -> bool:
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return sys.stdout.isatty()


def _format_progress_stats(progress: TaskExecutionProgress) -> str:
    return (
        f"upd={progress.updated} same={progress.unchanged} "
        f"skip={progress.skipped} err={progress.errors} "
        f"cancel={progress.canceled}"
    )


@contextmanager
def _llm_progress_callback():
    if not _is_progress_render_enabled():
        yield None
        return

    console = rich_get_console()
    progress_task_id: Any | None = None
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[stats]}"),
        console=console,
        transient=False,
    ) as progress_view:

        def update(progress: TaskExecutionProgress) -> None:
            nonlocal progress_task_id
            if progress.total <= 0:
                return
            if progress_task_id is None:
                progress_task_id = progress_view.add_task(
                    description=f"LLM {progress.task_name}",
                    total=progress.total,
                    completed=progress.completed,
                    stats=_format_progress_stats(progress),
                )
                return
            progress_view.update(
                progress_task_id,
                total=progress.total,
                completed=progress.completed,
                stats=_format_progress_stats(progress),
            )

        yield update

"""LLM command-line entrypoints and parser wiring."""

from __future__ import annotations

import argparse
import logging
import shlex
import shutil
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ankiops.config import (
    LLM_DIR,
    file_stem_to_deck_name,
    get_note_types_dir,
    require_initialized_collection_dir,
)
from ankiops.fs import FileSystemAdapter

from .config_loader import load_llm_task_catalog
from .model_registry import MODEL_REGISTRY_FILE_NAME
from .runner import list_jobs as list_llm_jobs
from .runner import plan_task, run_task, show_job

logger = logging.getLogger(__name__)


def _load_note_type_configs(note_types_dir: Path) -> list[Any]:
    return FileSystemAdapter().load_note_type_configs(note_types_dir)


def _format_field_list(fields: list[str]) -> str:
    return ", ".join(fields) if fields else "-"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _usage_error(message: str) -> None:
    logger.error(message)
    raise SystemExit(2)


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

    def _wrap_cell(value: str, width: int) -> list[str]:
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

    def _render_row(values: list[str]) -> list[str]:
        wrapped_cells = [
            _wrap_cell(values[index], widths[index]) for index in range(len(values))
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

    separator = ["-" * width for width in widths]
    rendered_lines = [*(_render_row(headers)), *(_render_row(separator))]
    for row in rows:
        rendered_lines.extend(_render_row(row))
    return rendered_lines


def _log_table(headers: list[str], rows: list[list[str]]) -> None:
    for line in _format_table(headers, rows):
        logger.info(line)


def configure_llm_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handler: Callable[[argparse.Namespace], None],
) -> None:
    """Register the ``ankiops llm`` command."""
    llm_parser = subparsers.add_parser(
        "llm",
        help="Plan/run LLM jobs and inspect LLM job history",
    )
    llm_parser.add_argument(
        "task_name",
        nargs="?",
        help="Task name to plan/run",
    )
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
    *,
    require_initialized_collection_dir_fn: Callable[[], Path] = (
        require_initialized_collection_dir
    ),
    get_note_types_dir_fn: Callable[[], Path] = get_note_types_dir,
    load_note_type_configs_fn: Callable[[Path], list[Any]] | None = None,
    load_llm_task_catalog_fn: Callable[..., Any] = load_llm_task_catalog,
    plan_task_fn: Callable[..., Any] = plan_task,
    run_task_fn: Callable[..., Any] = run_task,
    list_jobs_fn: Callable[..., Any] = list_llm_jobs,
    show_job_fn: Callable[..., Any] = show_job,
) -> None:
    """Status/plan/run/show for LLM jobs."""
    if load_note_type_configs_fn is None:
        load_note_type_configs_fn = _load_note_type_configs

    collection_dir = require_initialized_collection_dir_fn()
    task_name = getattr(args, "task_name", None)
    run_mode = bool(getattr(args, "run", False))
    job_id = getattr(args, "job_id", None)
    model_override = getattr(args, "model", None)
    deck_override = _normalize_deck_override(getattr(args, "deck", None))
    no_auto_commit = bool(getattr(args, "no_auto_commit", False))

    if job_id is not None:
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
            detail = show_job_fn(
                collection_dir=collection_dir,
                job_id=job_id,
            )
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        if detail is None:
            logger.error(f"Unknown LLM job '{job_id}'")
            raise SystemExit(1)

        logger.info(
            "Job %s — %s (%s / %s)",
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
        logger.info(
            "Summary: %s",
            detail.summary.format(),
        )
        logger.info(
            "Usage: %s",
            detail.summary.format_usage(),
        )
        logger.info(
            "Cost: %s",
            detail.summary.format_cost(),
        )
        if detail.fatal_error:
            logger.error("Fatal error: %s", detail.fatal_error)
        logger.info("")
        logger.info("Items:")
        if not detail.items:
            logger.info("  none")
        else:
            item_rows: list[list[str]] = []
            for item in detail.items:
                item_rows.append(
                    [
                        str(item.ordinal),
                        item.note_key or "unknown",
                        item.deck_name,
                        item.note_type or "unknown",
                        item.item_status.value,
                        str(item.attempts),
                        f"{_format_count(item.input_tokens)}/{_format_count(item.output_tokens)}",
                        f"{item.latency_ms / 1000:.2f}s",
                        _format_field_list(item.changed_fields),
                    ]
                )

            _log_table(
                [
                    "#",
                    "Note",
                    "Deck",
                    "Type",
                    "Final",
                    "Attempts",
                    "Tokens",
                    "Latency",
                    "Changed",
                ],
                item_rows,
            )

            for item in detail.items:
                if item.error_message:
                    logger.error("  #%d error=%s", item.ordinal, item.error_message)
        return

    if task_name is None:
        if run_mode:
            _usage_error("--run requires <task>.")
        if model_override is not None:
            _usage_error("--model requires <task>.")
        if deck_override is not None:
            _usage_error("--deck requires <task>.")
        if no_auto_commit:
            _usage_error("--no-auto-commit requires <task> --run.")

        note_type_configs = load_note_type_configs_fn(get_note_types_dir_fn())
        catalog = load_llm_task_catalog_fn(
            collection_dir,
            note_type_configs=note_type_configs,
        )
        logger.info("LLM config: %s", "OK" if not catalog.errors else "INVALID")
        logger.info("")
        logger.info("Tasks:")
        tasks = sorted(catalog.tasks_by_name.values(), key=lambda task: task.name)
        if tasks:
            task_rows: list[list[str]] = []
            for task in tasks:
                task_rows.append(
                    [
                        task.name,
                        str(task.model),
                        str(len(task.field_rules)),
                    ]
                )
            _log_table(
                ["Name", "Model", "Field rules"],
                task_rows,
            )
        else:
            logger.info("  none")

        logger.info("")
        logger.info("Recent jobs:")
        jobs = list_jobs_fn(collection_dir=collection_dir)
        if not jobs:
            logger.info("  none")
        else:
            job_rows: list[list[str]] = []
            for job in jobs:
                job_rows.append(
                    [
                        str(job.job_id),
                        job.task_name,
                        job.model,
                        job.status.value,
                        "yes" if job.persisted else "no",
                        job.created_at,
                    ]
                )
            _log_table(
                ["Job", "Task", "Model", "Status", "Persisted", "Created"],
                job_rows,
            )

        if catalog.errors:
            for message in catalog.errors.values():
                logger.error(message)
            raise SystemExit(1)
        return

    if no_auto_commit and not run_mode:
        _usage_error("--no-auto-commit requires --run.")

    if run_mode:
        try:
            result = run_task_fn(
                collection_dir=collection_dir,
                task_name=task_name,
                model_override=model_override,
                deck_override=deck_override,
                no_auto_commit=no_auto_commit,
            )
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        inspect_command = f"ankiops llm --job {result.job_id}"
        if result.failed:
            logger.error(
                "LLM job #%d failed with %d error(s)%s. Cost: %s. Inspect: %s",
                result.job_id,
                result.summary.errors,
                " (no updates persisted due to errors)"
                if not result.persisted and result.summary.updated > 0
                else "",
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
        return

    try:
        plan = plan_task_fn(
            collection_dir=collection_dir,
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
    if plan.prompt_path is not None:
        logger.info("Task prompt file: %s", plan.prompt_path)
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
        surface_rows: list[list[str]] = []
        for surface in plan.field_surface:
            surface_rows.append(
                [
                    surface.note_type,
                    str(surface.candidate_notes),
                    _format_field_list(surface.editable_fields),
                    _format_field_list(surface.read_only_fields),
                    _format_field_list(surface.hidden_fields),
                ]
            )
        _log_table(
            ["Type", "Candidates", "Editable", "Read-only", "Hidden"],
            surface_rows,
        )
    else:
        logger.info("  none")
    logger.info("")
    logger.info("Full prompt:")
    logger.info("%s", plan.format_full_prompt())
    logger.info("")
    logger.info("Request estimate: %s", _format_count(plan.requests_estimate))
    logger.info("Cost estimate (worst-case): %s", plan.format_cost_estimate())
    run_command = f"ankiops llm {plan.task_name} --run"
    if model_override:
        run_command = f"{run_command} --model {model_override}"
    if deck_override is not None:
        run_command = f"{run_command} --deck {shlex.quote(deck_override)}"
    logger.info("To run this task: %s", run_command)

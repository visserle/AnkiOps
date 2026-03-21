"""LLM command-line entrypoints and parser wiring."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ankiops.config import get_note_types_dir, require_initialized_collection_dir
from ankiops.fs import FileSystemAdapter

from .anthropic_models import supported_model_names
from .config_loader import load_llm_task_catalog
from .runner import list_jobs as list_llm_jobs
from .runner import plan_task, run_task, show_job

logger = logging.getLogger(__name__)


def _load_note_type_configs(note_types_dir: Path) -> list[Any]:
    return FileSystemAdapter().load_note_type_configs(note_types_dir)


def _format_field_list(fields: list[str]) -> str:
    return ",".join(fields) if fields else "-"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _usage_error(message: str) -> None:
    logger.error(message)
    raise SystemExit(2)


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
        help="Show one LLM job (numeric id, '-1', or 'latest')",
    )
    llm_parser.add_argument(
        "--model",
        choices=supported_model_names(),
        help="Override model class for this plan/run (opus, sonnet, haiku)",
    )
    llm_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip automatic git snapshot (only with <task> --run)",
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
    no_auto_commit = bool(getattr(args, "no_auto_commit", False))

    if job_id is not None:
        if task_name is not None:
            _usage_error("Cannot combine <task> with --job.")
        if run_mode:
            _usage_error("Cannot combine --run with --job.")
        if model_override is not None:
            _usage_error("--model requires <task>.")
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
            detail.model_name,
            detail.api_model,
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
        logger.info("Items:")
        for item in detail.items:
            note = item.note_key or "unknown"
            note_type = item.note_type or "unknown"
            logger.info(
                "  #%d note=%s deck=%s type=%s candidate=%s final=%s "
                "attempts=%d tokens=%s/%s latency=%0.2fs",
                item.ordinal,
                note,
                item.deck_name,
                note_type,
                item.candidate_status.value,
                item.final_status.value,
                item.attempts,
                _format_count(item.input_tokens),
                _format_count(item.output_tokens),
                item.latency_ms / 1000,
            )
            if item.changed_fields:
                logger.info("    changed_fields=%s", ",".join(item.changed_fields))
            if item.error_message:
                logger.error("    error=%s", item.error_message)
        return

    if task_name is None:
        if run_mode:
            _usage_error("--run requires <task>.")
        if model_override is not None:
            _usage_error("--model requires <task>.")
        if no_auto_commit:
            _usage_error("--no-auto-commit requires <task> --run.")

        note_type_configs = load_note_type_configs_fn(get_note_types_dir_fn())
        catalog = load_llm_task_catalog_fn(
            collection_dir,
            note_type_configs=note_type_configs,
        )
        logger.info("LLM config: %s", "OK" if not catalog.errors else "INVALID")
        logger.info("Tasks:")
        tasks = sorted(catalog.tasks_by_name.values(), key=lambda task: task.name)
        if tasks:
            for task in tasks:
                include = ",".join(task.decks.include)
                exclude = ",".join(task.decks.exclude) if task.decks.exclude else "-"
                logger.info(
                    "  - %s (model=%s, include=%s, exclude=%s, exceptions=%d)",
                    task.name,
                    task.model,
                    include,
                    exclude,
                    len(task.field_exceptions),
                )
        else:
            logger.info("  none")

        logger.info("Recent jobs:")
        jobs = list_jobs_fn(collection_dir=collection_dir)
        if not jobs:
            logger.info("  none")
        else:
            for job in jobs:
                logger.info(
                    "  - #%s %s (%s) status=%s persisted=%s created=%s",
                    job.job_id,
                    job.task_name,
                    job.model_name,
                    job.status.value,
                    "yes" if job.persisted else "no",
                    job.created_at,
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
                no_auto_commit=no_auto_commit,
            )
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        logger.info(
            "LLM job: %d",
            result.job_id,
        )
        if result.failed:
            logger.info("Actual cost: %s", result.summary.format_cost())
            logger.error(
                "LLM job finished with %d error(s)%s",
                result.summary.errors,
                " (atomic policy: no updates persisted)"
                if not result.persisted and result.summary.updated > 0
                else "",
            )
            raise SystemExit(1)
        logger.info("Actual cost: %s", result.summary.format_cost())
        return

    try:
        plan = plan_task_fn(
            collection_dir=collection_dir,
            task_name=task_name,
            model_override=model_override,
        )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    logger.info("Plan: %s (model=%s)", plan.task_name, plan.model)
    logger.info("Deck scope: %s", plan.deck_scope)
    logger.info("Serializer scope: %s", plan.serializer_scope)
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
    logger.info("Field surface:")
    if plan.field_surface:
        for surface in plan.field_surface:
            logger.info(
                "  %s  candidates=%d  editable=%s  read_only=%s  hidden=%s",
                surface.note_type,
                surface.candidate_notes,
                _format_field_list(surface.editable_fields),
                _format_field_list(surface.read_only_fields),
                _format_field_list(surface.hidden_fields),
            )
    else:
        logger.info("  none")
    logger.info("Request estimate: %s", _format_count(plan.requests_estimate))
    logger.info("Cost estimate (max): %s", plan.format_cost_estimate())
    run_command = f"ankiops llm {plan.task_name} --run"
    if model_override:
        run_command = f"{run_command} --model {model_override}"
    logger.info("To run this task: %s", run_command)

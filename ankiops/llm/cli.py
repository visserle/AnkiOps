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
from .runner import run_task, show_job

logger = logging.getLogger(__name__)


def _load_note_type_configs(note_types_dir: Path) -> list[Any]:
    return FileSystemAdapter().load_note_type_configs(note_types_dir)


def configure_llm_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    handler: Callable[[argparse.Namespace], None],
) -> None:
    """Register the ``ankiops llm`` command tree."""
    llm_parser = subparsers.add_parser(
        "llm",
        help="List and run configured LLM tasks, inspect LLM job history",
    )
    llm_subparsers = llm_parser.add_subparsers(
        dest="llm_action",
        required=False,
    )

    llm_list_parser = llm_subparsers.add_parser(
        "list",
        help="List configured LLM tasks",
    )
    llm_list_parser.set_defaults(handler=handler, llm_action="list")

    llm_run_parser = llm_subparsers.add_parser(
        "run",
        help="Run one configured LLM task",
    )
    llm_run_parser.add_argument(
        "task_name",
        help="Task name to run",
    )
    llm_run_parser.add_argument(
        "--model",
        choices=supported_model_names(),
        help="Override model class (opus, sonnet, haiku)",
    )
    llm_run_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    llm_run_parser.set_defaults(handler=handler, llm_action="run")

    llm_jobs_parser = llm_subparsers.add_parser(
        "jobs",
        help="Show recent LLM job runs",
    )
    llm_jobs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of jobs to show (default: 20)",
    )
    llm_jobs_parser.set_defaults(handler=handler, llm_action="jobs")

    llm_show_parser = llm_subparsers.add_parser(
        "show",
        help="Show details for one LLM job",
    )
    llm_show_parser.add_argument(
        "job_id",
        help="Job ID from 'ankiops llm jobs' (numeric id or 'latest')",
    )
    llm_show_parser.set_defaults(handler=handler, llm_action="show")

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
    run_task_fn: Callable[..., Any] = run_task,
    list_jobs_fn: Callable[..., Any] = list_llm_jobs,
    show_job_fn: Callable[..., Any] = show_job,
) -> None:
    """List/run LLM tasks and inspect LLM job history."""
    if load_note_type_configs_fn is None:
        load_note_type_configs_fn = _load_note_type_configs

    collection_dir = require_initialized_collection_dir_fn()
    action = getattr(args, "llm_action", None) or "list"

    if action == "list":
        note_type_configs = load_note_type_configs_fn(get_note_types_dir_fn())
        catalog = load_llm_task_catalog_fn(
            collection_dir,
            note_type_configs=note_type_configs,
        )
        tasks = sorted(catalog.tasks_by_name.values(), key=lambda task: task.name)
        if tasks:
            for task in tasks:
                include = ",".join(task.decks.include)
                exclude = ",".join(task.decks.exclude) if task.decks.exclude else "-"
                logger.info(
                    f"{task.name}  model={task.model}  include={include}  "
                    f"exclude={exclude}  exceptions={len(task.field_exceptions)}"
                )
        if catalog.errors:
            for message in catalog.errors.values():
                logger.error(message)
            raise SystemExit(1)
        return

    if action == "run":
        if not getattr(args, "task_name", None):
            logger.error("Task name is required (usage: ankiops llm run <task>)")
            raise SystemExit(2)
        try:
            result = run_task_fn(
                collection_dir=collection_dir,
                task_name=args.task_name,
                model_override=args.model,
                no_auto_commit=args.no_auto_commit,
            )
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        logger.info(
            "LLM job: %d",
            result.job_id,
        )
        if result.failed:
            logger.error(
                "LLM task finished with %d error(s)%s",
                result.summary.errors,
                " (atomic policy: no updates persisted)"
                if not result.persisted and result.summary.updated > 0
                else "",
            )
            raise SystemExit(1)
        return

    if action == "jobs":
        jobs = list_jobs_fn(collection_dir=collection_dir, limit=args.limit)
        if not jobs:
            logger.info("No LLM jobs found.")
            return
        for job in jobs:
            logger.info(
                "%s  task=%s  model=%s  status=%s  persisted=%s  created_at=%s",
                job.job_id,
                job.task_name,
                job.model_name,
                job.status.value,
                "yes" if job.persisted else "no",
                job.created_at,
            )
        return

    if action == "show":
        try:
            detail = show_job_fn(
                collection_dir=collection_dir,
                job_id=args.job_id,
            )
        except ValueError as error:
            logger.error(str(error))
            raise SystemExit(1) from error
        if detail is None:
            logger.error(f"Unknown LLM job '{args.job_id}'")
            raise SystemExit(1)

        logger.info(
            "Job %s  task=%s model=%s api_model=%s status=%s persisted=%s",
            detail.job_id,
            detail.task_name,
            detail.model_name,
            detail.api_model,
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
        for item in detail.items:
            note = item.note_key or "unknown"
            note_type = item.note_type or "unknown"
            logger.info(
                "  #%d note=%s deck=%s type=%s candidate=%s final=%s "
                "attempts=%d tokens=%d/%d latency_ms=%d",
                item.ordinal,
                note,
                item.deck_name,
                note_type,
                item.candidate_status.value,
                item.final_status.value,
                item.attempts,
                item.input_tokens,
                item.output_tokens,
                item.latency_ms,
            )
            if item.changed_fields:
                logger.info("    changed_fields=%s", ",".join(item.changed_fields))
            if item.error_message:
                logger.error("    error=%s", item.error_message)
        return

    logger.error(f"Unknown llm action '{action}'")
    raise SystemExit(2)

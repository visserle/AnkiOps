"""AI-specific CLI handlers and argument helpers."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from ankiops.ai import (
    AIConfigError,
    AIPaths,
    AIRequestError,
    AIResponseError,
    AIRuntimeOverrides,
    TaskConfigError,
    TaskExecutionError,
    TaskRunner,
    TaskRunOptions,
    build_async_editor,
    prepare_ai_run,
)
from ankiops.ai.model_config import (
    load_model_configs,
    provider_choices,
    resolve_runtime_config,
)
from ankiops.ai.types import TaskProgressUpdate
from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import ANKIOPS_DB, get_collection_dir
from ankiops.git import git_snapshot

logger = logging.getLogger(__name__)

_WARNING_MAX_CHARS = 180


class _TaskProgressLogger:
    """Throttled logger callback for long-running AI task progress."""

    def __init__(self, *, interval_seconds: float = 5.0):
        self._interval_seconds = interval_seconds
        self._last_emitted_at = 0.0
        self._last_warning_count = 0

    def __call__(self, update: TaskProgressUpdate) -> None:
        warning_count_increased = update.warning_count > self._last_warning_count
        is_final_chunk = (
            update.queued_chunks > 0
            and update.completed_chunks == update.queued_chunks
        )
        elapsed_since_emit = update.elapsed_seconds - self._last_emitted_at
        if (
            not warning_count_increased
            and not is_final_chunk
            and elapsed_since_emit < self._interval_seconds
        ):
            return
        self._last_warning_count = update.warning_count
        self._last_emitted_at = update.elapsed_seconds
        percent = (
            (update.completed_chunks / update.queued_chunks) * 100.0
            if update.queued_chunks > 0
            else 0.0
        )
        notes_per_second = (
            update.matched_notes / update.elapsed_seconds
            if update.elapsed_seconds > 0
            else 0.0
        )
        logger.info(
            "AI progress: "
            f"{update.completed_chunks}/{update.queued_chunks} chunk(s) "
            f"({percent:.0f}%), "
            f"in flight {update.in_flight}, "
            f"elapsed {update.elapsed_seconds:.1f}s — "
            f"{update.processed_notes} scanned, "
            f"{update.matched_notes} matched, "
            f"{update.changed_fields} changed, "
            f"{update.warning_count} warning(s), "
            f"{notes_per_second:.1f} notes/s"
        )


def _progress_enabled(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return sys.stderr.isatty()


def _format_warning_for_cli(
    warning: str,
    *,
    max_chars: int = _WARNING_MAX_CHARS,
) -> str:
    normalized = re.sub(r"\s+", " ", warning.replace("\t", " ").strip())
    if not normalized:
        return normalized

    hint = _warning_hint(normalized)
    if hint and "hint:" not in normalized.lower():
        normalized = f"{normalized}; {hint}"

    if ": " not in normalized:
        return _truncate_text(normalized, max_chars)

    prefix, detail = normalized.split(": ", 1)
    if "/" not in prefix:
        return _truncate_text(normalized, max_chars)

    prefix_with_sep = f"{prefix}: "
    detail_max_chars = max_chars - len(prefix_with_sep)
    if detail_max_chars <= 0:
        return _truncate_text(normalized, max_chars)
    return prefix_with_sep + _truncate_text(detail, detail_max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def _warning_hint(value: str) -> str | None:
    lowered = value.lower()
    if "unexpected field key" in lowered:
        return "hint: check task write_fields and returned patch keys."
    if "non-string value" in lowered:
        return "hint: all patched field values must be JSON strings."
    return None


def _log_performance_summary(started_at: float, result, client) -> None:
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    warning_count = len(result.warnings) + result.dropped_warnings
    logger.info(
        "AI: completed in "
        f"{elapsed_seconds:.2f}s — "
        f"{result.processed_notes} scanned, "
        f"{result.matched_notes} matched, "
        f"{result.changed_fields} field(s) changed, "
        f"{warning_count} warning(s), "
        f"{result.processed_decks} deck(s)"
    )

    metrics = client.metrics()
    attempts = metrics.http_attempts
    average_attempt_latency_ms = (
        (metrics.total_attempt_seconds / attempts) * 1000.0
        if attempts > 0
        else 0.0
    )
    retries_label = "retry" if metrics.retries == 1 else "retries"
    logger.info(
        "AI transport: "
        f"{metrics.logical_requests} logical request(s), "
        f"{attempts} HTTP attempt(s), "
        f"{metrics.retries} {retries_label}, "
        f"{average_attempt_latency_ms:.1f}ms avg attempt latency"
    )


def _positive_int(value: str) -> int:
    raw = value.strip()
    try:
        parsed = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _temperature_value(value: str) -> float:
    raw = value.strip()
    try:
        parsed = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 2") from error
    if not 0 <= parsed <= 2:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return parsed


def require_initialized_collection_dir() -> Path:
    """Return collection directory or exit if no local AnkiOps DB exists."""
    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)
    return collection_dir


def _runtime_overrides_from_args(args) -> AIRuntimeOverrides:
    return AIRuntimeOverrides(
        profile=args.profile,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        max_in_flight=args.max_in_flight,
        api_key=args.api_key,
    )


def run_ai_config(args):
    """Show the resolved AI runtime configuration from ai/models/*.yaml."""
    collection_dir = require_initialized_collection_dir()
    ai_paths = AIPaths.from_collection_dir(collection_dir)

    try:
        models_config = load_model_configs(ai_paths)
    except AIConfigError as error:
        logger.error(f"Invalid model profile configuration: {error}")
        raise SystemExit(1)

    overrides = _runtime_overrides_from_args(args)
    try:
        runtime = resolve_runtime_config(
            models_config,
            profile=overrides.profile,
            provider=overrides.provider,
            model=overrides.model,
            base_url=overrides.base_url,
            api_key_env=overrides.api_key_env,
            timeout_seconds=overrides.timeout_seconds,
            max_in_flight=overrides.max_in_flight,
            api_key=overrides.api_key,
        )
    except AIConfigError as error:
        logger.error(f"Invalid AI configuration: {error}")
        raise SystemExit(1)

    logger.info(f"Models config: {models_config.source_path}")
    logger.info(f"Default profile: {models_config.default_profile}")
    logger.info(f"Selected profile: {runtime.profile}")
    logger.info(f"AI provider: {runtime.provider}")
    logger.info(f"AI model: {runtime.model}")
    logger.info(f"AI base URL: {runtime.base_url}")
    logger.info(f"AI timeout: {runtime.timeout_seconds}s")
    logger.info(f"AI max in flight: {runtime.max_in_flight}")
    logger.info(f"API key env var: {runtime.api_key_env}")
    logger.info(f"API key available: {'yes' if runtime.api_key else 'no'}")


def run_ai_task(args):
    """Run task-driven inline JSON edits over serialized collection data."""
    if not args.task:
        logger.error("Missing required argument: --task")
        raise SystemExit(2)

    collection_dir = require_initialized_collection_dir()

    overrides = _runtime_overrides_from_args(args)
    try:
        task_config, runtime = prepare_ai_run(
            collection_dir=collection_dir,
            task_ref=args.task,
            overrides=overrides,
        )
    except TaskConfigError as error:
        logger.error(f"Invalid task configuration: {error}")
        raise SystemExit(1)
    except AIConfigError as error:
        logger.error(f"Invalid AI configuration: {error}")
        raise SystemExit(1)

    if args.temperature is not None:
        task_config = replace(task_config, temperature=args.temperature)

    if runtime.requires_api_key and not runtime.api_key:
        env_hint = f" in env var '{runtime.api_key_env}'" if runtime.api_key_env else ""
        logger.error(f"No API key found{env_hint}. Set it or pass --api-key.")
        raise SystemExit(1)

    resolved_batch_size = args.batch_size or task_config.batch_size
    scope = ", ".join(args.include_deck) if args.include_deck else "all decks"
    logger.info(
        "AI: running task "
        f"'{task_config.id}' "
        f"with profile '{runtime.profile}' "
        f"({runtime.provider}/{runtime.model})"
    )
    logger.info(
        "AI: batch "
        f"'{task_config.batch}' "
        f"(size {resolved_batch_size}), "
        f"max in flight {runtime.max_in_flight}, "
        f"temperature {task_config.temperature}"
    )
    logger.info(f"AI: scope {scope}")

    if not args.no_auto_commit:
        logger.debug("Creating pre-ai git snapshot")
        git_snapshot(collection_dir, "ai")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    serialized_data = serialize_collection(collection_dir)
    started_at = time.perf_counter()
    progress_callback = (
        _TaskProgressLogger() if _progress_enabled(args.progress) else None
    )

    client = build_async_editor(runtime)
    options = TaskRunOptions(
        include_decks=args.include_deck,
        batch_size=args.batch_size,
        max_in_flight=runtime.max_in_flight,
        progress_callback=progress_callback,
    )
    try:
        result = TaskRunner(client).run(
            serialized_data=serialized_data,
            task_config=task_config,
            options=options,
        )
    except (TaskExecutionError, AIRequestError, AIResponseError) as error:
        logger.error(f"AI task failed: {error}")
        raise SystemExit(1)

    _log_performance_summary(started_at, result, client)

    if args.include_deck and result.processed_decks == 0:
        logger.warning("No deck matched --include-deck filters.")
        return

    logger.info(f"AI changes ({result.changed_fields} field(s)):")

    for change in result.changes[:20]:
        logger.info(
            f"  {change.deck_name} [{change.note_key or 'new'}] {change.field_name}"
        )
    if len(result.changes) > 20:
        logger.info(f"  ... and {len(result.changes) - 20} more change(s)")

    warning_count = len(result.warnings) + result.dropped_warnings
    if warning_count:
        logger.warning(f"AI warnings ({warning_count}):")

    for warning in result.warnings[:20]:
        logger.warning(_format_warning_for_cli(warning))
    remaining_warnings = max(0, len(result.warnings) - 20) + result.dropped_warnings
    if remaining_warnings:
        logger.warning(f"... and {remaining_warnings} more warning(s)")

    if result.changed_fields == 0:
        logger.info("No changes to write.")
        return

    apply_payload = {
        "collection": serialized_data.get("collection", {}),
        "decks": result.changed_decks,
    }

    deserialize_collection_data(
        apply_payload,
        overwrite=True,
    )

    logger.info(f"Applied changes to {len(result.changed_decks)} deck(s).")


def add_ai_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Attach runtime override flags to an argparse parser."""
    parser.add_argument(
        "--profile",
        help="Model profile id from ai/models/*.yaml",
    )
    parser.add_argument(
        "--provider",
        choices=list(provider_choices()),
        help="Optional runtime provider override",
    )
    parser.add_argument(
        "--model",
        help="Optional runtime model override",
    )
    parser.add_argument(
        "--base-url",
        help="Optional runtime OpenAI-compatible base URL override",
    )
    parser.add_argument(
        "--api-key-env",
        help="Optional runtime API key env var override",
    )
    parser.add_argument(
        "--api-key",
        help="Optional runtime API key value",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        help="Optional runtime timeout override",
    )
    parser.add_argument(
        "--max-in-flight",
        type=_positive_int,
        help="Optional runtime max concurrent request override",
    )


def add_ai_task_args(parser: argparse.ArgumentParser) -> None:
    """Attach task execution flags to the ai task-run command."""
    parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    parser.add_argument(
        "--include-deck",
        "-d",
        action="append",
        default=[],
        help="Include a deck and all subdecks recursively (repeatable)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task file name/path from ai/tasks/ (required)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Override task batch size",
    )
    parser.add_argument(
        "--temperature",
        type=_temperature_value,
        default=None,
        help="Override task temperature (0 to 2)",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="Show periodic AI task progress logs (default: auto)",
    )

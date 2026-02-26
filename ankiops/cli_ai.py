"""AI-specific CLI handlers and argument helpers."""

from __future__ import annotations

import argparse
import logging
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
from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import ANKIOPS_DB, get_collection_dir

logger = logging.getLogger(__name__)


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

    logger.info(
        "AI task run "
        f"task='{task_config.id}' "
        f"profile='{runtime.profile}' "
        f"provider='{runtime.provider}' model='{runtime.model}' "
        f"batch='{task_config.batch}' "
        f"batch_size={args.batch_size or task_config.batch_size} "
        f"max_in_flight={runtime.max_in_flight} "
        f"temperature={task_config.temperature}"
    )

    serialized_data = serialize_collection(collection_dir)

    client = build_async_editor(runtime)
    options = TaskRunOptions(
        include_decks=args.include_deck,
        batch_size=args.batch_size,
        max_in_flight=runtime.max_in_flight,
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

    if args.include_deck and result.processed_decks == 0:
        logger.warning("No deck matched --include-deck filters.")
        return

    logger.info(
        "AI task processed "
        f"{result.matched_notes} matched note(s), "
        f"{result.processed_notes} scanned note(s), "
        f"across {result.processed_decks} deck(s)."
    )
    logger.info(f"AI task changed {result.changed_fields} field(s).")

    for change in result.changes[:20]:
        logger.info(
            f"  {change.deck_name} [{change.note_key or 'new'}] {change.field_name}"
        )
    if len(result.changes) > 20:
        logger.info(f"  ... and {len(result.changes) - 20} more change(s)")

    for warning in result.warnings[:20]:
        logger.warning(warning)
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

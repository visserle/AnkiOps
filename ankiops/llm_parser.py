"""Lightweight CLI parser for LLM commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ankiops.collection import LLM_DIR, MODEL_REGISTRY_FILE_NAME


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
        "--job", dest="job_id", help="Select one LLM job (numeric id or 'latest')"
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


def run_llm(args: argparse.Namespace) -> None:
    from ankiops.llm.commands import run_llm as run

    run(args)

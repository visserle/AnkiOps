"""Console logging and terminal-facing helpers."""

from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from rich import get_console as rich_get_console
from rich import reconfigure as rich_reconfigure
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.markup import escape as rich_escape
from rich.text import Text

from ankiops.anki import Anki

DEFAULT_QUIET_LOGGERS = (
    "urllib3.connectionpool",
    "openai",
    "httpx",
    "httpcore",
)
DEFAULT_TRACEBACK_SUPPRESS: tuple[str | ModuleType, ...] = (argparse,)

logger = logging.getLogger(__name__)


def clickable_path(file_path: Path | str) -> str:
    """Create Rich markup for a clickable file path."""
    path = Path(file_path).expanduser().resolve(strict=False)
    text = path.name if path.name else str(path)
    file_uri = path.as_uri()
    return f"[link={file_uri}]{rich_escape(text)}[/link]"


def print_line(value: str = "") -> None:
    rich_get_console().print(value, markup=False, highlight=False, soft_wrap=True)


def print_result(action: str, subject: str, summary: str) -> None:
    heading = Text()
    heading.append("✓ ", style="bold green")
    heading.append(f"{action.capitalize()} {subject}", style="bold")
    rich_get_console().print(heading)
    print_line(f"  {summary}")


def print_next_steps(steps: list[str]) -> None:
    if not steps:
        return
    print_line()
    rich_get_console().print(Text("Next", style="bold"))
    for step in steps:
        print_line(f"  {step}")


def print_error(value: str) -> None:
    message = Text("Error: ", style="bold red")
    message.append(value)
    Console(stderr=True).print(message, soft_wrap=True)


def configure_logging(
    stream_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    file_path: Path | str | None = None,
    stream: bool = True,
    ignore_libs: str | list[str] | None = None,
    tracebacks_suppress: tuple[str | ModuleType, ...] | None = None,
) -> None:
    """Configure the root logger for Rich console output and optional file logs."""
    root_handlers: list[str] = []
    active_levels: list[int] = []
    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "stream_verbose": {"format": "%(message)s"},
            "stream_compact": {"format": "%(message)s"},
            "file": {
                "format": "{asctime} | {levelname:8} | {name} | {message}",
                "style": "{",
            },
        },
        "handlers": {},
        "root": {"level": logging.WARNING, "handlers": root_handlers},
    }
    suppress_targets = (
        DEFAULT_TRACEBACK_SUPPRESS
        if tracebacks_suppress is None
        else tracebacks_suppress
    )

    if stream:
        verbose = stream_level <= logging.DEBUG
        console = rich_get_console()
        if console.file is not sys.stdout:
            rich_reconfigure(file=sys.stdout)
            console = rich_get_console()
        logging_config["handlers"]["rich_stream"] = {
            "class": "rich.logging.RichHandler",
            "level": stream_level,
            "console": console,
            "show_time": verbose,
            "show_level": True,
            "show_path": verbose,
            "enable_link_path": False,
            "rich_tracebacks": verbose,
            "tracebacks_suppress": suppress_targets,
            "markup": False,
            "highlighter": NullHighlighter(),
            "log_time_format": "%H:%M:%S",
            "formatter": "stream_verbose" if verbose else "stream_compact",
        }
        root_handlers.append("rich_stream")
        active_levels.append(stream_level)

    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "filename": str(file_path),
            "encoding": "utf-8",
            "formatter": "file",
        }
        root_handlers.append("file")
        active_levels.append(file_level)

    extra_ignored_libs: list[str] = []
    if ignore_libs:
        if isinstance(ignore_libs, str):
            extra_ignored_libs = [ignore_libs]
        else:
            extra_ignored_libs = list(ignore_libs)

    quiet_loggers = dict.fromkeys((*DEFAULT_QUIET_LOGGERS, *extra_ignored_libs))
    for lib in quiet_loggers:
        logging.getLogger(lib).setLevel(logging.WARNING)

    root_level = min(active_levels, default=logging.WARNING)
    logging_config["root"]["level"] = root_level
    logging_config["root"]["handlers"] = root_handlers
    logging.config.dictConfig(logging_config)


def connect_or_exit() -> Anki:
    """Verify an Anki HTTP connection is reachable; exit on failure."""
    anki = Anki()
    try:
        version = anki.get_version()
        logger.debug("Connected to Anki HTTP API (version %s)", version)
    except Exception as error:
        logger.error(
            "Error connecting to Anki. Make sure Anki is running and either "
            "AnkiOpsConnect or AnkiConnect is enabled."
        )
        logger.debug("Connection error details: %s", error)
        raise SystemExit(1)
    return anki

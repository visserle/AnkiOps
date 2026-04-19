"""Logging configuration for the root logger (Python >= 3.10)."""

import argparse
import logging
import logging.config
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from rich import get_console as rich_get_console
from rich import reconfigure as rich_reconfigure
from rich.highlighter import NullHighlighter

DEFAULT_QUIET_LOGGERS = (
    "urllib3.connectionpool",
    "openai",
    "httpx",
    "httpcore",
)
DEFAULT_TRACEBACK_SUPPRESS: tuple[str | ModuleType, ...] = (argparse,)


def format_changes(**counts: int) -> str:
    """Format non-zero change counts into a compact string.

    >>> format_changes(updated=3, created=1, deleted=0, skipped=96)
    '3 updated, 1 created'
    >>> format_changes(skipped=10)
    'no changes'
    """
    parts = []
    # 'total' is usually displayed separately (e.g. "X files - Y synced")
    # so we skip it in the compact change summary.
    for label_key, count_value in counts.items():
        if not count_value or label_key == "total":
            continue
        # Singularize nouns (e.g. "1 errors" → "1 error")
        label = (
            label_key[:-1]
            if count_value == 1 and label_key.endswith("s")
            else label_key
        )
        parts.append(f"{count_value} {label}")
    return ", ".join(parts) if parts else "no changes"


def clickable_path(file_path: Path | str, display_name: str | None = None) -> str:
    """Create a clickable terminal hyperlink for a file path.

    Uses OSC 8 escape sequences to create clickable links in modern terminals
    (VSCode, iTerm2, Terminal.app, Windows Terminal, GNOME Terminal, etc.).

    Args:
        file_path: Path object or string to make clickable
        display_name: Optional display text (defaults to filename only)

    Returns:
        String with OSC 8 escape codes for terminal hyperlinks
    """
    path = Path(file_path).expanduser()
    absolute_path = path.resolve(strict=False)
    text = display_name if display_name is not None else f"FILE {absolute_path}"

    # Respect NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return text

    # Build an RFC-compliant file URI so spaces/special chars are encoded
    # (e.g. " " -> %20, "#" -> %23).
    file_uri = absolute_path.as_uri()

    # OSC 8 format: \033]8;;FILE_URI\033\\TEXT\033]8;;\033\\
    return f"\033]8;;{file_uri}\033\\{text}\033]8;;\033\\"


def configure_logging(
    stream_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    file_path: Path | str | None = None,
    stream: bool = True,
    ignore_libs: str | list[str] | None = None,
    tracebacks_suppress: tuple[str | ModuleType, ...] | None = None,
) -> None:
    """
    Configures the root logger for console and file logging with the specified
    parameters.

    Parameters:
    - stream_level: The logging level for the stream handler.
    - file_level: The logging level for the file handler.
    - file_path: The path to the log file (if None, file logging is disabled).
    - stream: Whether to enable console logging (default is True).
    - ignore_libs: A list of library names to ignore in the logs.
    - tracebacks_suppress: Optional modules/paths to suppress in Rich tracebacks.

    Example usage in main script:
    >>> import logging
    >>> from log_config import configure_logging
    >>> configure_logging()
    >>> logging.debug("This is a debug message.")
    """

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

    # Console logging via Rich (single renderer for log + progress output)
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

    # FileHandler for file logging, added only if file path is provided
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

    # Set the logging level for ignored libraries to WARNING
    extra_ignored_libs: list[str] = []
    if ignore_libs:
        if isinstance(ignore_libs, str):
            extra_ignored_libs = [ignore_libs]
        else:
            extra_ignored_libs = list(ignore_libs)

    quiet_loggers = dict.fromkeys((*DEFAULT_QUIET_LOGGERS, *extra_ignored_libs))
    for lib in quiet_loggers:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Reconfigure root logging atomically using dictConfig.
    root_level = min(active_levels, default=logging.WARNING)
    logging_config["root"]["level"] = root_level
    logging_config["root"]["handlers"] = root_handlers
    logging.config.dictConfig(logging_config)


def close_root_logging() -> None:
    """
    Safely closes and removes all handlers associated with the root logger.

    Note that handlers typically do not require manual closing and removal,
    as Python's logging module automatically manages this process when the program
    terminates.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


def main():
    """Example usage of the configure_logging function."""
    configure_logging(
        stream_level=logging.DEBUG,
        file_level=10,
    )
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")

    # Ignore warnings from the 'ignored' library in this example
    configure_logging(ignore_libs=["ignored"])
    logger = logging.getLogger("ignored")
    logging.info("This info from root logger will be shown.")
    logger.info("This info from 'ignored' library will be ignored.")
    logger.warning("This warning from 'ignored' library will be shown.")


if __name__ == "__main__":
    main()

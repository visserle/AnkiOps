"""Logging configuration for the root logger (Python >= 3.10)."""

import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape as rich_escape

DEFAULT_QUIET_LOGGERS = (
    "urllib3.connectionpool",
    "openai",
    "httpx",
    "httpcore",
)

_RICH_CONSOLE: Console | None = None


class _RichSeverityMessageFormatter(logging.Formatter):
    """Format plain log messages with Rich markup by severity."""

    def __init__(self) -> None:
        super().__init__("%(message)s")

    def format(self, record: logging.LogRecord) -> str:
        message = rich_escape(super().format(record))
        if record.levelno >= logging.CRITICAL:
            return f"[bold red]{message}[/]"
        if record.levelno >= logging.ERROR:
            return f"[red]{message}[/]"
        if record.levelno >= logging.WARNING:
            return f"[orange3]{message}[/]"
        return message


def get_rich_console() -> Console:
    """Return a shared Rich console bound to the current stdout stream."""
    global _RICH_CONSOLE
    if _RICH_CONSOLE is None or _RICH_CONSOLE.file is not sys.stdout:
        _RICH_CONSOLE = Console(file=sys.stdout)
    return _RICH_CONSOLE


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

    Example usage in main script:
    >>> import logging
    >>> from log_config import configure_logging
    >>> configure_logging()
    >>> logging.debug("This is a debug message.")
    """

    handlers = []

    # Console logging via Rich (single renderer for log + progress output)
    if stream:
        verbose = stream_level <= logging.DEBUG
        stream_handler = RichHandler(
            console=get_rich_console(),
            show_time=verbose,
            show_level=verbose,
            show_path=verbose,
            enable_link_path=False,
            rich_tracebacks=verbose,
            markup=not verbose,
            log_time_format="%H:%M:%S",
        )
        stream_handler.setLevel(stream_level)
        if verbose:
            stream_handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
        else:
            stream_handler.setFormatter(_RichSeverityMessageFormatter())
        handlers.append(stream_handler)

    # FileHandler for file logging, added only if file path is provided
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            "{asctime} | {levelname:8} | {name} | {message}", style="{"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

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

    # Clear any previously added handlers from the root logger
    logging.getLogger().handlers = []

    # Set up the root logger configuration with the specified handlers
    logging.basicConfig(level=min(stream_level, file_level), handlers=handlers)


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

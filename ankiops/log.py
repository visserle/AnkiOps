"""Logging configuration for the root logger (Python >= 3.10)."""

import logging
import platform
import sys
from pathlib import Path


def format_changes(**counts: int) -> str:
    """Format non-zero change counts into a compact string.

    >>> format_changes(updated=3, created=1, deleted=0, skipped=96)
    '3 updated, 1 created'
    >>> format_changes(skipped=10)
    'no changes'
    """
    parts = []
    for k, v in counts.items():
        if not v:
            continue
        # Singularize nouns (e.g. "1 errors" → "1 error")
        label = k[:-1] if v == 1 and k.endswith("s") else k
        parts.append(f"{v} {label}")
    return ", ".join(parts) if parts else "no changes"


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

    # StreamHandler for console logging
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(stream_level)

        if stream_level <= logging.DEBUG:
            # Verbose format for debug mode
            stream_format = (
                "{asctime} | {color}{levelname:8}{reset} | {name} | {message}"
            )
            stream_formatter = ColoredFormatter(
                stream_format,
                style="{",
                datefmt="%H:%M:%S",
            )
        else:
            # Clean format for normal CLI use
            stream_formatter = CleanFormatter()

        stream_handler.setFormatter(stream_formatter)
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
    if ignore_libs:
        if isinstance(ignore_libs, str):
            ignore_libs = [ignore_libs]
        for lib in ignore_libs:
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


class Color:
    """A class for terminal color codes using ANSI escape sequences."""

    BLUE = "\033[36m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BOLD_RED = BOLD + RED + UNDERLINE
    END = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Logging Formatter class that adds colors and styles to log messages."""

    COLORS = {
        "DEBUG": Color.BLUE,
        "INFO": Color.GREEN,
        "WARNING": Color.YELLOW,
        "ERROR": Color.RED,
        "CRITICAL": Color.BOLD_RED,
        "RESET": Color.END,
    }

    # Enable ANSI escape sequences on Windows
    if platform.system() == "Windows":
        try:
            from colorama import just_fix_windows_console

            just_fix_windows_console()
        except ImportError:
            print("Colorama module not found, proceeding without colored logging.")
            # No colors, but include 'RESET' for consistency
            COLORS = {"RESET": ""}

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""
        super().__init__(*args, **kwargs)
        self.colors = ColoredFormatter.COLORS

    def format(self, record) -> str:
        """Format the specified record as text."""
        record.color = self.colors.get(record.levelname, "")
        record.reset = self.colors["RESET"]
        return super().format(record)


class CleanFormatter(logging.Formatter):
    """Minimal formatter for CLI output — message only, color for warnings/errors."""

    LEVEL_COLORS = {
        "WARNING": Color.YELLOW,
        "ERROR": Color.RED,
        "CRITICAL": Color.BOLD_RED,
    }

    def format(self, record) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        reset = Color.END if color else ""
        return f"{color}{record.getMessage()}{reset}"


def main():
    """Example usage of the configure_logging function."""
    configure_logging(
        stream_level=logging.DEBUG,
        file_level=10,
        file_path="debug.log",
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

"""Configuration for Anki to Markdown conversion."""

import logging
import re
from pathlib import Path
from urllib.parse import unquote

from ankiops.models import ANKIOPS_KEY_FIELD

logger = logging.getLogger(__name__)

ANKIOPS_DB = ".ankiops.db"
NOTE_TYPES_DIR = "note_types"
LOCAL_MEDIA_DIR = "media"
LLM_DIR = "llm"
LLM_DB_FILENAME = ".llm.db"

NOTE_SEPARATOR = "\n\n---\n\n"  # changing the whitespace might lead to issues

NOTE_KEY_PATTERN = re.compile(r"^\s*<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->\s*$")
NOTE_TYPE_PATTERN = re.compile(r"^\s*<!--\s*note_type:\s*.*?\s*-->\s*$")
CODE_FENCE_PATTERN = re.compile(r"^(```|~~~)")
LABEL_CANDIDATE_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*:)(?:\s|$)")
RESERVED_NOTE_FIELD_NAMES = frozenset({ANKIOPS_KEY_FIELD.name})


def sanitize_filename(deck_name: str) -> str:
    """Convert deck name to a safe filename (``::`` → ``__``).

    Raises ValueError for invalid characters or Windows reserved names.
    """
    invalid = [char for char in r'/\\?*|"<>' if char in deck_name and char != ":"]
    if invalid:
        raise ValueError(
            f"Deck name '{deck_name}' contains invalid filename characters: "
            f"{invalid}\nPlease rename the deck in Anki to remove these."
        )

    base = deck_name.split("::")[0].upper()
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if base in windows_reserved:
        raise ValueError(
            f"Deck name '{deck_name}' starts with Windows reserved name "
            f"'{base}'.\nPlease rename the deck in Anki."
        )

    return deck_name.replace("::", "__")


def deck_name_to_file_stem(deck_name: str) -> str:
    """Encode an Anki deck name to a reversible markdown filename stem.

    - ``::`` (subdeck separator) becomes ``__`` for readability.
    - Literal ``__`` is escaped as ``%5F%5F`` to avoid ambiguity.
    - Literal ``%`` is escaped first so decoding is lossless.
    - Path separators (``/`` and ``\\``) are escaped to keep files at root.
    """
    return (
        deck_name.replace("%", "%25")
        .replace("__", "%5F%5F")
        .replace("::", "__")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )


def file_stem_to_deck_name(file_stem: str) -> str:
    """Decode a markdown filename stem back to its deck name."""
    return unquote(file_stem.replace("__", "::"))


def get_collection_dir() -> Path:
    """Get the collection directory path.

    Development mode (pyproject.toml in cwd): ./collection
    Otherwise: current working directory
    """
    pyproject = Path.cwd() / "pyproject.toml"
    if pyproject.exists():
        try:
            if 'name = "ankiops"' in pyproject.read_text():
                return Path.cwd() / "collection"
        except OSError:
            pass
    return Path.cwd()


def require_collection_dir(active_profile: str | None = None) -> Path:
    """Return the collection directory, or exit if not initialized.

    Also exits if the active profile doesn't match (when provided).
    """
    from ankiops.db import SQLiteDbAdapter

    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)

    if active_profile is None:
        return collection_dir

    db = SQLiteDbAdapter.open(collection_dir)
    try:
        expected_profile = db.get_profile_name()
        if expected_profile is None:
            logger.error(
                f"Collection in {collection_dir} has no linked profile. "
                "Run 'ankiops init' to re-link."
            )
            raise SystemExit(1)

        if expected_profile != active_profile:
            logger.error(
                f"Profile mismatch: collection in {collection_dir} is linked to "
                f"'{expected_profile}', but Anki has '{active_profile}' "
                f"open. Switch profiles in Anki, or re-run "
                f"'ankiops init' to re-link."
            )
            raise SystemExit(1)
    finally:
        db.close()

    return collection_dir

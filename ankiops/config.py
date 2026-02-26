"""Configuration for Anki to Markdown conversion."""

import logging
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

ANKIOPS_DB = ".ankiops.db"
NOTE_TYPES_DIR = "note_types"
AI_DIR = "ai"
AI_PROMPTS_DIR = "prompts"
LOCAL_MEDIA_DIR = "media"

NOTE_SEPARATOR = "\n\n---\n\n"  # changing the whitespace might lead to issues


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
    """
    return deck_name.replace("%", "%25").replace("__", "%5F%5F").replace("::", "__")


def file_stem_to_deck_name(file_stem: str) -> str:
    """Decode a markdown filename stem back to its deck name."""
    return unquote(file_stem.replace("__", "::"))


def _is_development_mode() -> bool:
    """Check if running from the AnkiOps source tree."""
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        return 'name = "ankiops"' in pyproject.read_text()
    except OSError:
        return False


def get_collection_dir() -> Path:
    """Get the collection directory path.

    Development mode (pyproject.toml in cwd): ./collection
    Otherwise: current working directory
    """
    if _is_development_mode():
        return Path.cwd() / "collection"
    return Path.cwd()


def get_note_types_dir() -> Path:
    """Get the standard note types directory path."""
    return get_collection_dir() / NOTE_TYPES_DIR


def get_ai_dir() -> Path:
    """Get the standard AI directory path."""
    return get_collection_dir() / AI_DIR


def get_ai_prompts_dir() -> Path:
    """Get the standard AI prompts directory path."""
    return get_ai_dir() / AI_PROMPTS_DIR


def require_collection_dir(active_profile: str) -> Path:
    """Return the collection directory, or exit if not initialized.

    Also exits if the active profile doesn't match.
    """
    from ankiops.db import SQLiteDbAdapter

    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)

    db = SQLiteDbAdapter.load(collection_dir)
    expected_profile = db.get_profile_name()
    if expected_profile and expected_profile != active_profile:
        logger.error(
            f"Profile mismatch: collection in {collection_dir} is linked to "
            f"'{expected_profile}', but Anki has '{active_profile}' "
            f"open. Switch profiles in Anki, or re-run "
            f"'ankiops init' to re-link."
        )
        db.close()
        raise SystemExit(1)
    db.close()

    return collection_dir

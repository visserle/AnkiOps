"""Configuration for Anki to Markdown conversion."""

import configparser
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER_FILE = ".ankiops"



NOTE_SEPARATOR = "\n\n---\n\n"  # changing the whitespace might lead to issues

COMMON_FIELDS = [
    ("Extra", "E:", False),
    ("More", "M:", False),
    ("Source", "S:", False),
    ("AI Notes", "AI:", False),
]

# Each note type has a list of field mappings, where each mapping is a tuple of:
# (field_name, prefix, required)
NOTE_TYPES = {
    "AnkiOpsQA": {
        "field_mappings": [
            ("Question", "Q:", True),
            ("Answer", "A:", True),
            *COMMON_FIELDS,
        ],
    },
    "AnkiOpsReversed": {
        "field_mappings": [
            ("Front", "F:", True),
            ("Back", "B:", True),
            *COMMON_FIELDS,
        ],
    },
    "AnkiOpsCloze": {
        "field_mappings": [
            ("Text", "T:", True),
            *COMMON_FIELDS,
        ],
    },
    "AnkiOpsInput": {
        "field_mappings": [
            ("Question", "Q:", True),
            ("Input", "I:", True),
            *COMMON_FIELDS,
        ],
    },
    "AnkiOpsChoice": {
        "field_mappings": [
            ("Question", "Q:", True),
            ("Choice 1", "C1:", True),
            ("Choice 2", "C2:", False),
            ("Choice 3", "C3:", False),
            ("Choice 4", "C4:", False),
            ("Choice 5", "C5:", False),
            ("Choice 6", "C6:", False),
            ("Choice 7", "C7:", False),
            ("Choice 8", "C8:", False),
            ("Answer", "A:", True),
            *COMMON_FIELDS,
        ],
    },
}

SUPPORTED_NOTE_TYPES = list(NOTE_TYPES.keys())


def sanitize_filename(deck_name: str) -> str:
    """Convert deck name to a safe filename (``::`` → ``__``).

    Raises ValueError for invalid characters or Windows reserved names.
    """
    invalid = [c for c in r'/\\?*|"<>' if c in deck_name and c != ":"]
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
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if base in windows_reserved:
        raise ValueError(
            f"Deck name '{deck_name}' starts with Windows reserved name "
            f"'{base}'.\nPlease rename the deck in Anki."
        )

    return deck_name.replace("::", "__")


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


def _read_marker(marker: Path) -> configparser.ConfigParser:
    """Read and return the parsed marker file."""
    config = configparser.ConfigParser()
    config.read(marker)
    return config


def require_collection_dir(active_profile: str) -> Path:
    """Return the collection directory, or exit if not initialized or profile mismatches."""
    collection_dir = get_collection_dir()
    marker = collection_dir / MARKER_FILE
    if not marker.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)

    config = _read_marker(marker)
    expected_profile = config.get("ankiops", "profile", fallback=None)
    if expected_profile and expected_profile != active_profile:
        logger.error(
            f"Profile mismatch: collection in {collection_dir} is linked to "
            f"'{expected_profile}', but Anki has '{active_profile}' "
            f"open. Switch profiles in Anki, or re-run "
            f"'ankiops init' to re-link."
        )
        raise SystemExit(1)

    return collection_dir




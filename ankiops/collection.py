"""Collection paths, initialization, and local file tree shape."""

from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

from ankiops.console import clickable_path
from ankiops.note_types import eject_default_note_types

logger = logging.getLogger(__name__)

ANKIOPS_DB = ".ankiops.db"
NOTE_TYPES_DIR = "note_types"
LOCAL_MEDIA_DIR = "media"
LLM_DIR = "llm"
LLM_DB_FILENAME = ".llm.db"


def sanitize_filename(deck_name: str) -> str:
    """Convert deck name to a safe filename (``::`` -> ``__``)."""
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
    """Encode an Anki deck name to a reversible Markdown filename stem."""
    if "_::" in deck_name or "::_" in deck_name:
        raise ValueError(
            f"Ambiguous deck name '{deck_name}': do not place '_' next to "
            "the '::' subdeck separator."
        )
    return (
        deck_name.replace("%", "%25")
        .replace("__", "%5F%5F")
        .replace("::", "__")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )


def file_stem_to_deck_name(file_stem: str) -> str:
    """Decode a Markdown filename stem back to its deck name."""
    return unquote(file_stem.replace("__", "::"))


def deck_name_in_scope(
    deck_name: str,
    *,
    deck: str | None,
    no_subdecks: bool,
) -> bool:
    """Return whether `deck_name` is selected by a deck/subdeck scope."""
    deck_filter = deck.strip() if isinstance(deck, str) else None
    if deck_filter is None:
        return True
    if no_subdecks:
        return deck_name == deck_filter
    return deck_name == deck_filter or deck_name.startswith(f"{deck_filter}::")


def get_collection_dir() -> Path:
    """Get the collection directory path."""
    pyproject = Path.cwd() / "pyproject.toml"
    if pyproject.exists():
        try:
            if 'name = "ankiops"' in pyproject.read_text():
                return Path.cwd() / "collection"
        except OSError:
            pass
    return Path.cwd()


def require_collection_dir(active_profile: str | None = None) -> Path:
    """Return the collection directory, or exit if not initialized."""
    from ankiops.sync.state import SyncState

    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)

    if active_profile is None:
        return collection_dir

    sync_state = SyncState.open(collection_dir)
    try:
        expected_profile = sync_state.get_profile_name()
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
        sync_state.close()

    return collection_dir


def initialize_collection(profile: str) -> Path:
    """Initialize the current directory as an AnkiOps collection."""
    from ankiops.llm.jobs import LlmJobStore
    from ankiops.sync.state import SyncState

    collection_dir = get_collection_dir()
    collection_dir.mkdir(parents=True, exist_ok=True)

    sync_state = SyncState.open(collection_dir)
    sync_state.set_profile_name(profile)
    sync_state.close()
    llm_db = LlmJobStore.open(collection_dir)
    llm_db.close()

    (collection_dir / LOCAL_MEDIA_DIR).mkdir(exist_ok=True)
    _setup_vscode_settings(collection_dir)
    _setup_gitignore(collection_dir)
    _setup_git(collection_dir)
    _setup_llm_configs(collection_dir)
    eject_default_note_types(collection_dir / NOTE_TYPES_DIR)

    return collection_dir


def create_tutorial(collection_dir: Path) -> Path:
    """Copy the tutorial Markdown file to the collection directory."""
    tutorial_dst = collection_dir / f"{deck_name_to_file_stem('AnkiOps Tutorial')}.md"

    try:
        ref = resources.files("ankiops.tutorial").joinpath("AnkiOps Tutorial.md")
        tutorial_dst.write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(
            f"Tutorial file created: {clickable_path(tutorial_dst)}",
            extra={"markup": True},
        )

        img_ref = resources.files("ankiops.tutorial").joinpath("sync_arrows.png")
        img_dst = collection_dir / LOCAL_MEDIA_DIR / "sync_arrows.png"
        img_dst.write_bytes(img_ref.read_bytes())
        logger.info(
            f"Tutorial image created: {clickable_path(img_dst)}",
            extra={"markup": True},
        )

    except Exception as error:
        logger.warning(f"Could not create tutorial file: {error}")

    return tutorial_dst


def _setup_vscode_settings(collection_dir: Path) -> None:
    """Create/update .vscode/settings.json with markdown paste destination."""
    vscode_dir = collection_dir / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    settings_path = vscode_dir / "settings.json"

    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    settings.update(
        {
            "[markdown]": {
                "editor.formatOnSave": False,
                "editor.wordWrap": "wordWrapColumn",
                "editor.wordWrapColumn": 80,
                "editor.stickyScroll.enabled": False,
                "diffEditor.ignoreTrimWhitespace": True,
            },
            "markdown.preview.breaks": True,
            "markdown.preview.doubleClickToSwitchToEditor": True,
            "markdown.preview.markEditorSelection": True,
            "markdown.copyFiles.destination": {"**/*.md": f"{LOCAL_MEDIA_DIR}/"},
        }
    )
    settings_path.write_text(json.dumps(settings, indent=4) + "\n")


def _setup_gitignore(collection_dir: Path) -> None:
    """Ensure local database files are in .gitignore."""
    gitignore_path = collection_dir / ".gitignore"

    content = ""
    if gitignore_path.exists():
        content = gitignore_path.read_text()

    entries = [
        ".DS_Store",
        ANKIOPS_DB,
        f"{ANKIOPS_DB}-shm",
        f"{ANKIOPS_DB}-wal",
    ]
    missing = [entry for entry in entries if entry not in content]
    if missing:
        if content and not content.endswith("\n"):
            content += "\n"
        for entry in missing:
            content += f"{entry}\n"
        gitignore_path.write_text(content)
        logger.debug("Added local DB files to .gitignore")


def _setup_git(collection_dir: Path) -> None:
    """Ensure the collection directory is inside a git repository."""
    from ankiops.git import CollectionGit

    if CollectionGit(collection_dir).init_repo():
        logger.info(f"Initialized git repository in {collection_dir}")


def _eject_llm_configs(collection_dir: Path) -> None:
    from ankiops.llm.models import (
        MODEL_REGISTRY_FILE_NAME,
        SYSTEM_PROMPT_FILE_NAME,
    )

    llm_dir = collection_dir / LLM_DIR
    llm_dir.mkdir(parents=True, exist_ok=True)
    llm_resources = resources.files("ankiops.llm.resources")

    for resource_name in (MODEL_REGISTRY_FILE_NAME, SYSTEM_PROMPT_FILE_NAME):
        destination = llm_dir / resource_name
        if destination.exists():
            continue
        destination.write_text(
            llm_resources.joinpath(resource_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    for resource in llm_resources.iterdir():
        if (
            not resource.is_file()
            or resource.name == MODEL_REGISTRY_FILE_NAME
            or Path(resource.name).suffix not in {".yaml", ".yml"}
        ):
            continue
        destination = llm_dir / resource.name
        if destination.exists():
            continue
        destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")


def _setup_llm_configs(collection_dir: Path) -> None:
    _eject_llm_configs(collection_dir)

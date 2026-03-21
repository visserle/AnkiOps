"""Collection initialization for AnkiOps."""

import json
import logging
import subprocess
from pathlib import Path

from ankiops.config import (
    ANKIOPS_DB,
    LLM_DB_FILENAME,
    LLM_DIR,
    LOCAL_MEDIA_DIR,
    NOTE_TYPES_DIR,
    deck_name_to_file_stem,
    get_collection_dir,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.db import LlmDbAdapter
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)


def _setup_vscode_settings(collection_dir: Path):
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

    settings["markdown.preview.breaks"] = True
    settings["markdown.copyFiles.destination"] = {"**/*.md": f"{LOCAL_MEDIA_DIR}/"}
    settings_path.write_text(json.dumps(settings, indent=4) + "\n")


def _setup_gitignore(collection_dir: Path):
    """Ensure local database files are in .gitignore."""
    gitignore_path = collection_dir / ".gitignore"

    content = ""
    if gitignore_path.exists():
        content = gitignore_path.read_text()

    entries = [
        ANKIOPS_DB,
        f"{ANKIOPS_DB}-shm",
        f"{ANKIOPS_DB}-wal",
        f"{LLM_DIR}/{LLM_DB_FILENAME}",
        f"{LLM_DIR}/{LLM_DB_FILENAME}-shm",
        f"{LLM_DIR}/{LLM_DB_FILENAME}-wal",
    ]
    missing = [entry for entry in entries if entry not in content]
    if missing:
        if content and not content.endswith("\n"):
            content += "\n"
        for entry in missing:
            content += f"{entry}\n"
        gitignore_path.write_text(content)
        logger.debug("Added local DB files to .gitignore")


def _setup_git(collection_dir: Path):
    """Ensure the collection directory is inside a git repository.

    If it's already part of a repo (e.g. in development mode), this is a no-op.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=collection_dir,
        capture_output=True,
    )
    if result.returncode == 0:
        return  # already inside a git repo

    subprocess.run(
        ["git", "init"],
        cwd=collection_dir,
        capture_output=True,
        check=True,
    )
    logger.info(f"Initialized git repository in {collection_dir}")


def _eject_llm_configs(collection_dir: Path) -> None:
    from importlib import resources

    llm_dir = collection_dir / LLM_DIR
    llm_dir.mkdir(parents=True, exist_ok=True)

    system_prompt_src = resources.files("ankiops.llm").joinpath("system_prompt.md")
    system_prompt_dst = llm_dir / "system_prompt.md"
    if not system_prompt_dst.exists():
        system_prompt_dst.write_text(
            system_prompt_src.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    tasks_dir = llm_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    packaged_tasks_dir = resources.files("ankiops.llm").joinpath("tasks")
    for resource in packaged_tasks_dir.iterdir():
        if not resource.is_file() or resource.suffix not in {".yaml", ".yml"}:
            continue
        destination = tasks_dir / resource.name
        if destination.exists():
            continue
        destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")

    prompts_dir = llm_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    packaged_prompts_dir = resources.files("ankiops.llm").joinpath("prompts")
    for resource in packaged_prompts_dir.iterdir():
        if not resource.is_file() or resource.suffix != ".md":
            continue
        destination = prompts_dir / resource.name
        if destination.exists():
            continue
        destination.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")


def _setup_llm_configs(collection_dir: Path) -> None:
    _eject_llm_configs(collection_dir)


def initialize_collection(profile: str) -> Path:
    """Initialize the current directory as an AnkiOps collection.

    Creates the collection directory (if needed), initializes the database,
    creates the local media directory, and configures VSCode settings.
    Idempotent — safe to run multiple times.
    """
    collection_dir = get_collection_dir()
    collection_dir.mkdir(parents=True, exist_ok=True)

    db = SQLiteDbAdapter.open(collection_dir)
    db.set_profile_name(profile)
    db.close()
    llm_db = LlmDbAdapter.open(collection_dir)
    llm_db.close()

    (collection_dir / LOCAL_MEDIA_DIR).mkdir(exist_ok=True)
    _setup_vscode_settings(collection_dir)
    _setup_gitignore(collection_dir)
    _setup_git(collection_dir)
    _setup_llm_configs(collection_dir)

    # Eject built-in note types
    fs = FileSystemAdapter()
    fs.eject_builtin_note_types(collection_dir / NOTE_TYPES_DIR)

    return collection_dir


def create_tutorial(collection_dir: Path) -> Path:
    """Copy the tutorial markdown file to the collection directory."""
    from importlib import resources

    tutorial_dst = collection_dir / f"{deck_name_to_file_stem('AnkiOps Tutorial')}.md"

    try:
        # Python 3.9+ style
        ref = resources.files("ankiops.tutorial").joinpath("AnkiOps Tutorial.md")
        tutorial_dst.write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"Tutorial file created: {clickable_path(tutorial_dst)}")

        # Copy the image
        img_ref = resources.files("ankiops.tutorial").joinpath("sync_arrows.png")
        img_dst = collection_dir / LOCAL_MEDIA_DIR / "sync_arrows.png"
        img_dst.write_bytes(img_ref.read_bytes())
        logger.info(f"Tutorial image created: {clickable_path(img_dst)}")

    except Exception as error:
        logger.warning(f"Could not create tutorial file: {error}")

    return tutorial_dst

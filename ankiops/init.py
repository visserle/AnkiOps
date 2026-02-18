"""Collection initialization for AnkiOps."""

import configparser
import json
import logging
import platform
import subprocess
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR, MARKER_FILE, get_collection_dir
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)


def _setup_marker(collection_dir: Path, profile: str):
    """Write the .ankiops marker file with the active profile name."""
    marker = collection_dir / MARKER_FILE
    config = configparser.ConfigParser()
    config["ankiops"] = {
        "profile": profile,
    }
    with open(marker, "w") as f:
        f.write("# AnkiOps collection \u2014 do not delete this file.\n\n")
        config.write(f)


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

    settings["markdown.copyFiles.destination"] = {"**/*.md": f"{LOCAL_MEDIA_DIR}/"}
    settings_path.write_text(json.dumps(settings, indent=4) + "\n")


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


def initialize_collection(profile: str) -> Path:
    """Initialize the current directory as an AnkiOps collection.

    Creates the collection directory (if needed), writes the marker file,
    creates the local media directory, and configures VSCode settings.
    Idempotent — safe to run multiple times.
    """
    collection_dir = get_collection_dir()
    collection_dir.mkdir(parents=True, exist_ok=True)

    _setup_marker(collection_dir, profile)
    (collection_dir / LOCAL_MEDIA_DIR).mkdir(exist_ok=True)
    _setup_vscode_settings(collection_dir)
    _setup_git(collection_dir)

    return collection_dir


def create_tutorial(collection_dir: Path) -> Path:
    """Copy the tutorial markdown file to the collection directory."""
    from importlib import resources

    tutorial_dst = collection_dir / "AnkiOps Tutorial.md"

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

    except Exception as e:
        logger.warning(f"Could not create tutorial file: {e}")

    return tutorial_dst

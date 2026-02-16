"""Collection initialization for AnkiOps."""

import configparser
import json
import logging
import platform
import subprocess
from pathlib import Path

from ankiops.config import MARKER_FILE, get_collection_dir
from ankiops.log import clickable_path

logger = logging.getLogger(__name__)


def _setup_marker(collection_dir: Path, profile: str, media_dir: str):
    """Write the .ankiops marker file with the active profile name and media path."""
    marker = collection_dir / MARKER_FILE
    config = configparser.ConfigParser()
    config["ankiops"] = {
        "profile": profile,
        "media_dir": media_dir,
    }
    with open(marker, "w") as f:
        f.write("# AnkiOps collection \u2014 do not delete this file.\n\n")
        config.write(f)


def _is_junction(path: Path) -> bool:
    """Check if a path is a Windows directory junction."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        return attrs != -1 and bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)
    except Exception:
        return False


def _create_junction(link: Path, target: Path) -> bool:
    """Create a Windows directory junction. Returns True on success."""
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _setup_media_symlink(collection_dir: Path, media_dir: str):
    """Create a 'media' symlink/junction in the collection dir pointing to Anki media.

    On macOS/Linux, creates a symbolic link.
    On Windows, tries a symlink first (requires Developer Mode or admin privileges),
    then falls back to a directory junction.
    """
    link = collection_dir / "media"
    target = Path(media_dir)
    is_windows = platform.system() == "Windows"

    # Check if link already exists and is correct
    if link.is_symlink() or _is_junction(link):
        try:
            if link.resolve() == target.resolve():
                return  # already correct
        except OSError:
            pass
        link.unlink()
    elif link.exists():
        link.unlink()

    # Try symlink first (works on Unix, and Windows with Developer Mode)
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if not is_windows:
            raise  # On Unix, symlinks should work

    # Windows fallback: try directory junction
    if _create_junction(link, target):
        return

    # Neither worked — warn the user
    logger.warning(
        f"Could not create media link at {link}. "
        "On Windows, enable Developer Mode or run as administrator to create symlinks. "
        "Without this link, pasting images into the VS Code markdown editor will not "
        "save them directly to the Anki media folder."
    )


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

    settings["markdown.copyFiles.destination"] = {"**/*.md": "media/AnkiOpsMedia/"}
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


def initialize_collection(profile: str, media_dir: str) -> Path:
    """Initialize the current directory as an AnkiOps collection.

    Creates the collection directory (if needed), writes the marker file,
    sets up the media symlink, and configures VSCode settings.
    Idempotent — safe to run multiple times.
    """
    collection_dir = get_collection_dir()
    collection_dir.mkdir(parents=True, exist_ok=True)

    _setup_marker(collection_dir, profile, media_dir)
    _setup_media_symlink(collection_dir, media_dir)
    (collection_dir / "media" / "AnkiOpsMedia").mkdir(exist_ok=True)
    _setup_vscode_settings(collection_dir)
    _setup_git(collection_dir)

    return collection_dir


def create_tutorial(collection_dir: Path) -> Path:
    """Copy the tutorial markdown file to the collection directory."""
    from importlib import resources

    tutorial_dst = collection_dir / "AnkiOps Tutorial.md"

    try:
        # Python 3.9+ style
        ref = resources.files("ankiops.data").joinpath("AnkiOps Tutorial.md")
        tutorial_dst.write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"Tutorial file created: {clickable_path(tutorial_dst)}")
    except Exception as e:
        logger.warning(f"Could not create tutorial file: {e}")

    return tutorial_dst

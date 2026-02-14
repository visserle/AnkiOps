"""Automatic git snapshots before sync operations."""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def git_snapshot(collection_dir: Path, label: str) -> bool:
    """Commit all pending changes in the collection directory.

    Returns True if a commit was created, False otherwise.
    Never raises — logs warnings on failure so sync can proceed.
    """
    try:
        # Check if we're inside a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=collection_dir,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.debug("Not a git repository, skipping auto-commit")
            return False

        # Stage all changes in the collection directory
        subprocess.run(
            ["git", "add", "-A", "."],
            cwd=collection_dir,
            capture_output=True,
            check=True,
        )

        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=collection_dir,
            capture_output=True,
        )
        if result.returncode == 0:
            logger.debug("Working tree clean, skipping auto-commit")
            return False

        # Commit the snapshot
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = f"AnkiOps: pre-{label} snapshot ({timestamp})"
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=collection_dir,
            capture_output=True,
            check=True,
        )
        logger.info(f"Auto-committed snapshot before {label}")
        return True

    except subprocess.CalledProcessError as e:
        logger.warning(f"Auto-commit failed: {e}")
        return False
    except FileNotFoundError:
        logger.debug("Git not found, skipping auto-commit")
        return False

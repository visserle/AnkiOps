"""Git operations for an AnkiOps collection."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ankiops.sources import COLLAB_BRANCH, SyncSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionGit:
    collection_dir: Path

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        logger.debug("git %s", " ".join(args))
        return subprocess.run(
            ["git", *args],
            cwd=self.collection_dir,
            text=True,
            capture_output=True,
            check=check,
        )

    def is_repo(self) -> bool:
        return self.run(["rev-parse", "--git-dir"], check=False).returncode == 0

    def ensure_repo(self, message: str) -> None:
        if not self.is_repo():
            raise ValueError(message)

    def init_repo(self) -> bool:
        if self.is_repo():
            return False
        self.run(["init"])
        return True

    def head(self) -> str | None:
        result = self.run(["rev-parse", "--verify", "HEAD"], check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def ensure_clean_index(self, message: str) -> None:
        if self.cached_diff_exists():
            raise ValueError(message)

    def cached_diff_exists(self, rel_paths: list[str] | None = None) -> bool:
        args = ["diff", "--cached", "--quiet"]
        if rel_paths:
            args.extend(["--", *rel_paths])
        result = self.run(args, check=False)
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 1

    def tracked(self, rel_path: str) -> bool:
        return (
            self.run(
                ["ls-files", "--error-unmatch", "--", rel_path],
                check=False,
            ).returncode
            == 0
        )

    def commit_paths(self, paths: list[Path], message: str) -> None:
        rel_paths = self._tracked_or_existing_paths(paths)
        if not rel_paths:
            return
        self.run(["add", "-A", "--", *rel_paths])
        if self.cached_diff_exists(rel_paths):
            self.run(["commit", "-m", message, "--", *rel_paths])

    def commit_publish_move(
        self,
        *,
        touched_paths: list[Path],
        source_paths: list[Path],
        message: str,
    ) -> None:
        source_set = set(source_paths)
        add_paths = [
            self.rel_path(path)
            for path in touched_paths
            if path.exists() and path not in source_set
        ]
        if add_paths:
            self.run(["add", "-A", "--", *add_paths])

        source_rel_paths = [self.rel_path(path) for path in source_paths]
        if source_rel_paths:
            self.run(
                ["rm", "--cached", "-f", "--ignore-unmatch", "--", *source_rel_paths],
            )

        if self.cached_diff_exists():
            self.run(["commit", "-m", message])

    def rollback_to(self, initial_head: str | None) -> None:
        current_head = self.head()
        if current_head == initial_head or current_head is None:
            return
        if initial_head is None:
            self.run(["update-ref", "-d", "HEAD"])
        else:
            self.run(["reset", "--mixed", initial_head])

    def delete_branch_if_exists(self, branch: str | None) -> None:
        if branch is not None:
            self.run(["branch", "-D", branch], check=False)

    def unstage_or_untrack(self, paths: list[str]) -> None:
        if self.head() is not None:
            self.run(["reset", "HEAD", "--", *paths])
        else:
            self.run(["rm", "-r", "--cached", "--ignore-unmatch", "--", *paths])

    def subtree_add(self, source: SyncSource) -> None:
        self._run_subtree(source, "add")

    def subtree_pull(self, source: SyncSource) -> None:
        self._run_subtree(source, "pull")

    def subtree_split(self, source: SyncSource) -> str:
        branch = f"ankiops-{source.source_id.replace('/', '-')}-{uuid4().hex[:8]}"
        self.run(
            [
                "subtree",
                "split",
                "--prefix",
                self.source_prefix(source),
                "-b",
                branch,
            ],
        )
        return branch

    def remote_exists(self, url: str) -> bool:
        return self.run(["ls-remote", url], check=False).returncode == 0

    def push_ref(
        self,
        url: str,
        source_ref: str,
        target_ref: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        return self.run(["push", url, f"{source_ref}:{target_ref}"], check=check)

    def source_prefix(self, source: SyncSource) -> str:
        return self.rel_path(source.root)

    def rel_path(self, path: Path) -> str:
        return str(path.relative_to(self.collection_dir))

    def _tracked_or_existing_paths(self, paths: list[Path]) -> list[str]:
        rel_paths = []
        seen = set()
        for path in paths:
            rel_path = self.rel_path(path)
            if rel_path in seen:
                continue
            if path.exists() or self.tracked(rel_path):
                rel_paths.append(rel_path)
                seen.add(rel_path)
        return rel_paths

    def _run_subtree(self, source: SyncSource, action: str) -> None:
        if source.github_url is None:
            raise ValueError(f"Cannot derive GitHub URL for {source.display_name}")
        self.run(
            [
                "subtree",
                action,
                "--prefix",
                self.source_prefix(source),
                source.github_url,
                COLLAB_BRANCH,
            ],
        )


def git_snapshot(collection_dir: Path, label: str) -> bool:
    """Commit all pending changes in the collection directory.

    Returns True if a commit was created, False otherwise.
    Never raises; logs warnings on failure so sync can proceed.
    """
    repo = CollectionGit(collection_dir)
    try:
        if not repo.is_repo():
            logger.debug("Not a git repository, skipping auto-commit")
            return False

        repo.run(["add", "-A", "."])
        if not repo.cached_diff_exists():
            logger.debug("Working tree clean, skipping auto-commit")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        repo.run(["commit", "-m", f"AnkiOps: pre-{label} snapshot ({timestamp})"])
        logger.info(f"Auto-committed snapshot before {label}")
        return True

    except subprocess.CalledProcessError as error:
        logger.warning(f"Auto-commit failed: {error}")
        return False
    except FileNotFoundError:
        logger.debug("Git not found, skipping auto-commit")
        return False

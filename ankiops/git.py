"""Git operations for an AnkiOps collection."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ankiops.deck_sources import SHARED_BRANCH, DeckSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollectionGit:
    collection_dir: Path

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
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

    def refresh_index(self) -> None:
        self.run(["update-index", "-q", "--refresh"], check=False)

    def status_lines(self, pathspec: list[str]) -> list[str]:
        result = self.run(
            ["status", "--short", "--untracked-files=all", "--", *pathspec]
        )
        return result.stdout.splitlines()

    def tracked(self, rel_path: str) -> bool:
        return (
            self.run(
                ["ls-files", "--error-unmatch", "--", rel_path],
                check=False,
            ).returncode
            == 0
        )

    def commit_paths(self, paths: list[Path], message: str) -> bool:
        rel_paths = self._tracked_or_existing_paths(paths)
        if not rel_paths:
            return False
        self.run(["add", "-A", "--", *rel_paths])
        if self.cached_diff_exists(rel_paths):
            self.run(["commit", "-m", message, "--", *rel_paths])
            return True
        return False

    def commit_create_move(
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

    def subtree_add(self, source: DeckSource) -> None:
        self._run_subtree(
            source,
            "add",
            f"AnkiOps: add {source.source_id} from GitHub",
        )

    def subtree_pull(self, source: DeckSource) -> bool:
        old_head = self.head()
        stashed = self._stash_non_source_changes(source)
        try:
            self._run_subtree(
                source,
                "pull",
                f"AnkiOps: update {source.source_id} from GitHub",
            )
        finally:
            if stashed:
                self.run(["stash", "pop", "--index", "stash@{0}"])
        return self.head() != old_head

    def split_subtree(self, source: DeckSource) -> str:
        result = self.run(["subtree", "split", "--prefix", self.source_prefix(source)])
        split_sha = result.stdout.strip()
        if not split_sha:
            raise ValueError(f"Could not split {source.source_id}")
        return split_sha

    def create_temp_branch(self, source: DeckSource, commit_sha: str) -> str:
        branch = f"ankiops-{source.source_id.replace('/', '-')}-{uuid4().hex[:8]}"
        self.run(["branch", branch, commit_sha])
        return branch

    def fetch_source_head(self, source: DeckSource) -> str:
        if source.github_url is None:
            raise ValueError(f"Cannot derive GitHub URL for {source.display_name}")
        self.run(["fetch", source.github_url, SHARED_BRANCH])
        return self.run(["rev-parse", "FETCH_HEAD"]).stdout.strip()

    def has_subtree_metadata(self, source: DeckSource) -> bool:
        prefix = self.source_prefix(source)
        result = self.run(
            [
                "log",
                "-1",
                "--format=%B",
                f"--grep=^git-subtree-dir: {prefix}/*$",
                "HEAD",
            ],
            check=False,
        )
        if result.returncode != 0:
            return False
        trailers = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"git-subtree-mainline", "git-subtree-split"}:
                trailers[key] = value.strip()
        if set(trailers) != {"git-subtree-mainline", "git-subtree-split"}:
            return False
        return all(
            self.run(["cat-file", "-e", f"{sha}^{{commit}}"], check=False).returncode
            == 0
            for sha in trailers.values()
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self.run(
            ["merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    def has_common_ancestor(self, left: str, right: str) -> bool:
        result = self.run(["merge-base", left, right], check=False)
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    def trees_equal(self, left: str, right: str) -> bool:
        left_tree = self.run(["rev-parse", f"{left}^{{tree}}"]).stdout.strip()
        right_tree = self.run(["rev-parse", f"{right}^{{tree}}"]).stdout.strip()
        return left_tree == right_tree

    def rejoin_subtree(
        self,
        source: DeckSource,
        split_sha: str,
        subject: str,
    ) -> None:
        head = self.head()
        if head is None:
            raise ValueError("Cannot rejoin a subtree without a git commit.")

        prefix = self.source_prefix(source)
        source_tree = self.run(["rev-parse", f"{head}:{prefix}"]).stdout.strip()
        split_tree = self.run(["rev-parse", f"{split_sha}^{{tree}}"]).stdout.strip()
        if source_tree != split_tree:
            raise ValueError(
                f"Cannot rejoin {source.source_id}: split tree does not match "
                "the shared source."
            )

        head_tree = self.run(["rev-parse", f"{head}^{{tree}}"]).stdout.strip()
        trailers = "\n".join(
            [
                f"git-subtree-dir: {prefix}",
                f"git-subtree-mainline: {head}",
                f"git-subtree-split: {split_sha}",
            ]
        )
        commit = self.run(
            [
                "commit-tree",
                head_tree,
                "-p",
                head,
                "-p",
                split_sha,
                "-m",
                subject,
                "-m",
                trailers,
            ]
        ).stdout.strip()
        self.run(["update-ref", "-m", subject, "HEAD", commit, head])

    def remote_exists(self, url: str) -> bool:
        return self.run(["ls-remote", url], check=False).returncode == 0

    def push_ref(
        self,
        url: str,
        source_ref: str,
        target_ref: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["push", url, f"{source_ref}:{target_ref}"], check=check)

    def source_prefix(self, source: DeckSource) -> str:
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

    def _run_subtree(self, source: DeckSource, action: str, message: str) -> None:
        if source.github_url is None:
            raise ValueError(f"Cannot derive GitHub URL for {source.display_name}")
        self.refresh_index()
        self.run(
            [
                "subtree",
                action,
                "--prefix",
                self.source_prefix(source),
                "--message",
                message,
                source.github_url,
                SHARED_BRANCH,
            ],
        )

    def _stash_non_source_changes(self, source: DeckSource) -> bool:
        prefix = self.source_prefix(source)
        pathspec = [".", f":(exclude){prefix}", f":(exclude){prefix}/**"]
        status = self.run(
            ["status", "--porcelain=v1", "--untracked-files=all", "--", *pathspec]
        )
        if not status.stdout:
            return False
        self.run(
            [
                "stash",
                "push",
                "--include-untracked",
                "--message",
                f"AnkiOps: preserve private changes while updating {source.source_id}",
                "--",
                *pathspec,
            ]
        )
        return True


def git_snapshot(
    collection_dir: Path,
    *,
    action: str,
    paths: Sequence[Path],
) -> bool:
    """Commit pending changes for the supplied collection paths.

    Returns True if a commit was created, False otherwise.
    Never raises; logs warnings on failure so sync can proceed.
    """
    repo = CollectionGit(collection_dir)
    try:
        if not repo.is_repo():
            logger.debug("Not a git repository, skipping auto-commit")
            return False

        rel_paths = repo._tracked_or_existing_paths(list(paths))
        if not rel_paths:
            logger.debug("No scoped paths found, skipping auto-commit")
            return False

        repo.run(["add", "-A", "--", *rel_paths])
        if not repo.cached_diff_exists(rel_paths):
            logger.debug("Working tree clean, skipping auto-commit")
            return False

        repo.run(
            [
                "commit",
                "-m",
                f"AnkiOps: snapshot before {action}",
                "--",
                *rel_paths,
            ]
        )
        logger.info(f"Auto-committed snapshot before {action}")
        return True

    except subprocess.CalledProcessError as error:
        logger.warning(f"Auto-commit failed: {error}")
        return False
    except FileNotFoundError:
        logger.debug("Git not found, skipping auto-commit")
        return False

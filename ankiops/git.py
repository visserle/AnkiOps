"""Repository-scoped Git operations."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitRepository:
    """Run Git commands in one repository directory."""

    root: Path

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        logger.debug("git %s (in %s)", " ".join(args), self.root)
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=check,
        )

    @classmethod
    def clone(
        cls, url: str, target: Path, *, remote: str = "upstream"
    ) -> "GitRepository":
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--origin", remote, url, str(target)],
            text=True,
            capture_output=True,
            check=True,
        )
        return cls(target)

    def is_repo(self) -> bool:
        result = self.run(["rev-parse", "--show-toplevel"], check=False)
        if result.returncode != 0:
            return False
        try:
            return Path(result.stdout.strip()).resolve() == self.root.resolve()
        except OSError:
            return False

    def ensure_repo(self, message: str) -> None:
        if not self.is_repo():
            raise ValueError(message)

    def init_repo(self, *, initial_branch: str = "main") -> bool:
        if self.is_repo():
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        self.run(["init", "-b", initial_branch])
        return True

    def head(self) -> str | None:
        result = self.run(["rev-parse", "--verify", "HEAD"], check=False)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def tree(self, ref: str = "HEAD") -> str | None:
        result = self.run(["rev-parse", f"{ref}^{{tree}}"], check=False)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def trees_equal(self, left: str, right: str) -> bool:
        left_tree = self.tree(left)
        right_tree = self.tree(right)
        return (
            left_tree is not None and right_tree is not None and left_tree == right_tree
        )

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self.run(
            ["merge-base", "--is-ancestor", ancestor, descendant],
            check=False,
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    def status_lines(self, pathspec: list[str] | None = None) -> list[str]:
        args = ["status", "--short", "--untracked-files=all"]
        if pathspec:
            args.extend(["--", *pathspec])
        return self.run(args).stdout.splitlines()

    def tracked(self, rel_path: str) -> bool:
        return (
            self.run(
                ["ls-files", "--error-unmatch", "--", rel_path], check=False
            ).returncode
            == 0
        )

    def cached_diff_exists(self, rel_paths: list[str] | None = None) -> bool:
        args = ["diff", "--cached", "--quiet"]
        if rel_paths:
            args.extend(["--", *rel_paths])
        result = self.run(args, check=False)
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 1

    def checkpoint(self, message: str) -> str | None:
        if not self.status_lines():
            return None
        self.run(["add", "-A"])
        if not self.cached_diff_exists():
            return None
        self.run(["commit", "-m", message])
        return self.head()

    def commit_paths(self, paths: list[Path], message: str) -> bool:
        rel_paths = self._tracked_or_existing_paths(paths)
        if not rel_paths:
            return False
        self.run(["add", "-A", "--", *rel_paths])
        if not self.cached_diff_exists(rel_paths):
            return False
        self.run(["commit", "-m", message, "--", *rel_paths])
        return True

    def default_branch(self, remote: str = "upstream") -> str:
        symbolic = self.run(
            ["symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD"],
            check=False,
        )
        if symbolic.returncode == 0 and symbolic.stdout.strip():
            return symbolic.stdout.strip().split("/", 1)[1]
        for candidate in ("main", "master"):
            if (
                self.run(
                    ["show-ref", "--verify", f"refs/remotes/{remote}/{candidate}"],
                    check=False,
                ).returncode
                == 0
            ):
                return candidate
        raise ValueError(f"Cannot determine the default branch for remote {remote}.")

    def checkout_or_create_branch(self, branch: str, start_ref: str) -> None:
        exists = (
            self.run(
                ["show-ref", "--verify", f"refs/heads/{branch}"], check=False
            ).returncode
            == 0
        )
        if exists:
            self.run(["checkout", branch])
        else:
            self.run(["checkout", "-b", branch, start_ref])

    def fetch(self, remote: str = "upstream") -> None:
        self.run(["fetch", "--prune", remote])

    def integrate(self, ref: str, message: str) -> bool:
        old_head = self.head()
        self.run(["merge", "--no-edit", "-m", message, ref])
        return self.head() != old_head

    def unmerged_paths(self) -> list[str]:
        result = self.run(["diff", "--name-only", "--diff-filter=U", "-z"], check=False)
        return [path for path in result.stdout.split("\0") if path]

    def reset_hard(self, ref: str) -> None:
        self.run(["reset", "--hard", ref])

    def remote_url(self, remote: str) -> str | None:
        result = self.run(["remote", "get-url", remote], check=False)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def set_remote(self, remote: str, url: str) -> None:
        if self.remote_url(remote) is None:
            self.run(["remote", "add", remote, url])
        else:
            self.run(["remote", "set-url", remote, url])

    def push(self, remote: str, source_ref: str, branch: str) -> None:
        self.run(["push", remote, f"{source_ref}:refs/heads/{branch}"])

    def delete_remote_branch(self, remote: str, branch: str) -> None:
        self.run(["push", remote, "--delete", branch], check=False)

    def remote_branch_sha(self, remote: str, branch: str) -> str | None:
        result = self.run(
            ["ls-remote", "--heads", remote, f"refs/heads/{branch}"],
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return result.stdout.split()[0]

    def rel_path(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def _tracked_or_existing_paths(self, paths: list[Path]) -> list[str]:
        rel_paths = []
        for path in paths:
            rel_path = self.rel_path(path)
            if rel_path not in rel_paths and (path.exists() or self.tracked(rel_path)):
                rel_paths.append(rel_path)
        return rel_paths


def git_snapshot(
    collection_root: Path,
    *,
    action: str,
    paths: Sequence[Path],
) -> bool:
    """Commit pending changes for explicit paths in one repository."""
    collection_git = GitRepository(collection_root)
    try:
        collection_git.ensure_repo("AnkiOps collections require a Git repository.")
        rel_paths = collection_git._tracked_or_existing_paths(list(paths))
        if not rel_paths:
            return False
        collection_git.run(["add", "-A", "--", *rel_paths])
        if not collection_git.cached_diff_exists(rel_paths):
            return False
        collection_git.run(
            [
                "commit",
                "-m",
                f"AnkiOps: snapshot before {action}",
                "--",
                *rel_paths,
            ]
        )
        logger.info("Auto-committed snapshot before %s", action)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError(
            f"Could not create the required Git checkpoint before {action}: {error}"
        ) from error

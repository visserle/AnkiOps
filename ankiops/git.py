"""Repository-scoped Git operations."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_URL_CREDENTIALS = re.compile(r"(?P<scheme>https?://)[^/@\s'\"]+@", re.IGNORECASE)
_AUTHORIZATION_VALUE = re.compile(
    r"(?P<label>\bauthorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s'\"]+",
    re.IGNORECASE,
)
_LABELED_CREDENTIAL = re.compile(
    r"(?P<label>\b(?:access[_-]?token|password|passwd|token)\s*[:=]\s*)"
    r"[^\s&'\"]+",
    re.IGNORECASE,
)
_GITHUB_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)


class GitCommandError(subprocess.CalledProcessError):
    """A failed Git command with a useful, caller-facing diagnostic."""

    def __str__(self) -> str:
        detail = _concise_git_detail(self.stderr or self.stdout)
        message = f"Git command failed with exit {self.returncode}"
        return f"{message}: {detail}" if detail else message


class _GitCompletedProcess(subprocess.CompletedProcess[str]):
    def check_returncode(self) -> None:
        if self.returncode:
            raise GitCommandError(
                self.returncode,
                self.args,
                output=self.stdout,
                stderr=self.stderr,
            )


@dataclass(frozen=True)
class GitPathChange:
    status: str
    paths: tuple[str, ...]


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
        command = ["git", *args]
        safe_command = [_redact_git_text(part) for part in command]
        logger.debug("%s (in %s)", " ".join(safe_command), self.root)
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                capture_output=True,
                check=check,
            )
        except subprocess.CalledProcessError as error:
            safe_stdout = _redact_git_text(error.stdout)
            safe_stderr = _redact_git_text(error.stderr)
            detail = _concise_git_detail(safe_stderr or safe_stdout)
            logger.debug(
                "Git command failed with exit %s: %s",
                error.returncode,
                detail or "no diagnostic output",
            )
            raise GitCommandError(
                error.returncode,
                safe_command,
                output=safe_stdout,
                stderr=safe_stderr,
            ) from error
        if result.returncode == 0:
            return result
        safe_result = _GitCompletedProcess(
            safe_command,
            result.returncode,
            stdout=_redact_git_text(result.stdout),
            stderr=_redact_git_text(result.stderr),
        )
        logger.debug(
            "Git command failed with exit %s: %s",
            safe_result.returncode,
            _concise_git_detail(safe_result.stderr or safe_result.stdout)
            or "no diagnostic output",
        )
        if check:
            safe_result.check_returncode()
        return safe_result

    @classmethod
    def clone(
        cls,
        url: str,
        target: Path,
        *,
        remote: str = "upstream",
        anonymous: bool = False,
    ) -> "GitRepository":
        target.parent.mkdir(parents=True, exist_ok=True)
        args = ["git"]
        if anonymous:
            args.extend(["-c", "credential.helper="])
        args.extend(["clone", "--origin", remote, url, str(target)])
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"} if anonymous else None
        subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=True,
            env=env,
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

    def create_commit(self, tree: str, parent: str | None, message: str) -> str:
        """Create a commit object without moving the current branch."""
        args = ["commit-tree", tree]
        if parent is not None:
            args.extend(["-p", parent])
        args.extend(["-m", message])
        result = self.run(args)
        return result.stdout.strip()

    def worktree_matches(self, ref: str) -> bool:
        """Compare the final non-ignored worktree to a ref without writing objects."""
        untracked = self.run(
            ["ls-files", "--others", "--exclude-standard", "-z"]
        ).stdout
        if untracked:
            return False
        result = self.run(["diff", "--quiet", "--no-ext-diff", ref, "--"], check=False)
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    def ref_sha(self, ref: str) -> str | None:
        result = self.run(["rev-parse", "--verify", ref], check=False)
        return result.stdout.strip() or None if result.returncode == 0 else None

    def update_ref(self, ref: str, value: str) -> None:
        self.run(["update-ref", ref, value])

    def delete_ref(self, ref: str) -> None:
        self.run(["update-ref", "-d", ref], check=False)

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
        args = [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ]
        if pathspec:
            args.extend(["--", *pathspec])
        fields = self.run(args).stdout.split("\0")
        lines = []
        index = 0
        while index < len(fields) and fields[index]:
            entry = fields[index]
            status = entry[:2]
            path = entry[3:]
            index += 1
            if any(marker in status for marker in ("R", "C")):
                old_path = fields[index]
                index += 1
                path = f"{old_path} -> {path}"
            lines.append(f"{status} {path}")
        return lines

    def diff_name_status(
        self, base_ref: str, target_ref: str | None = None
    ) -> list[GitPathChange]:
        args = ["diff", "--name-status", "--find-renames", "-z", base_ref]
        if target_ref is not None:
            args.append(target_ref)
        fields = self.run(args).stdout.split("\0")
        changes = []
        index = 0
        while index < len(fields) and fields[index]:
            status = fields[index]
            index += 1
            path_count = 2 if status.startswith(("R", "C")) else 1
            paths = tuple(fields[index : index + path_count])
            index += path_count
            changes.append(GitPathChange(status, paths))
        return changes

    def diff_paths(self, base_ref: str, target_ref: str | None = None) -> list[str]:
        return [
            path
            for change in self.diff_name_status(base_ref, target_ref)
            for path in change.paths
        ]

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

    def push_force_with_lease(
        self,
        remote: str,
        source_ref: str,
        branch: str,
        expected_sha: str,
    ) -> None:
        self.run(
            [
                "push",
                f"--force-with-lease=refs/heads/{branch}:{expected_sha}",
                remote,
                f"{source_ref}:refs/heads/{branch}",
            ]
        )

    def delete_remote_branch_with_lease(
        self, remote: str, branch: str, expected_sha: str
    ) -> None:
        self.run(
            [
                "push",
                f"--force-with-lease=refs/heads/{branch}:{expected_sha}",
                remote,
                f":refs/heads/{branch}",
            ]
        )

    def remote_branch_sha(self, remote: str, branch: str) -> str | None:
        result = self.run(
            ["ls-remote", "--heads", remote, f"refs/heads/{branch}"],
            check=False,
        )
        if result.returncode != 0:
            result.check_returncode()
        if not result.stdout.strip():
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


def _redact_git_text(value: str | None) -> str:
    if not value:
        return ""
    redacted = _URL_CREDENTIALS.sub(r"\g<scheme><redacted>@", value)
    redacted = _AUTHORIZATION_VALUE.sub(r"\g<label><redacted>", redacted)
    redacted = _LABELED_CREDENTIAL.sub(r"\g<label><redacted>", redacted)
    return _GITHUB_TOKEN.sub("<redacted>", redacted)


def _concise_git_detail(output: str | None) -> str:
    detail = " ".join(_redact_git_text(output).split())
    return f"{detail[:497]}..." if len(detail) > 500 else detail


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
        staged_paths = [
            path
            for path in collection_git.run(
                ["diff", "--cached", "--name-only", "-z", "--", *rel_paths]
            ).stdout.split("\0")
            if path
        ]
        if not staged_paths:
            return False
        collection_git.run(
            [
                "commit",
                "-m",
                f"AnkiOps: snapshot before {action}",
                "--",
                *staged_paths,
            ]
        )
        logger.info("Auto-committed snapshot before %s", action)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError(
            f"Could not create the required Git checkpoint before {action}: {error}"
        ) from error

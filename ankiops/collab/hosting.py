"""Authenticated GitHub operations for collab repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ankiops.collab.errors import RepositoryCollisionError


def _is_fork_of(repository: dict[str, Any] | None, upstream_slug: str) -> bool:
    parent = (repository or {}).get("parent") or {}
    parent_slug = parent.get("full_name")
    return bool(
        (repository or {}).get("fork")
        and isinstance(parent_slug, str)
        and parent_slug.casefold() == upstream_slug.casefold()
    )


@dataclass(frozen=True)
class PullRequestInfo:
    url: str
    state: str
    title: str
    head_sha: str


@dataclass(frozen=True)
class GitHubHost:
    cwd: Path

    def _gh(
        self, args: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        gh = shutil.which("gh")
        if gh is None:
            raise ValueError(
                "GitHub CLI is required. Install gh, then run: gh auth login"
            )
        return subprocess.run(
            [gh, *args],
            cwd=self.cwd,
            text=True,
            capture_output=True,
            check=check,
        )

    def ensure_authenticated(self) -> None:
        result = self._gh(["auth", "status"], check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ValueError(
                "GitHub CLI is not authenticated. Run: gh auth login"
                + (f". Details: {detail}" if detail else "")
            )

    def repo_info(self, slug: str) -> dict[str, Any] | None:
        result = self._gh(["api", f"repos/{slug}"], check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            if "HTTP 404" in detail or "Not Found" in detail:
                return None
            raise ValueError(
                f"Could not inspect GitHub repository {slug}: "
                f"{detail or 'unknown error'}"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"GitHub returned invalid repository state for {slug}."
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"GitHub returned invalid repository state for {slug}.")
        return value

    def create_repo(self, slug: str) -> None:
        self.ensure_authenticated()
        result = self._gh(
            ["repo", "create", slug, "--public"],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(
                f"Could not create GitHub repository {slug}: {detail}. "
                "Retrying is safe."
            )

    def login(self) -> str:
        result = self._gh(["api", "user", "--jq", ".login"], check=False)
        login = result.stdout.strip()
        if result.returncode != 0 or not login:
            raise ValueError("GitHub CLI is not authenticated. Run: gh auth login")
        return login

    def publish_target(self, upstream_slug: str) -> tuple[str, str]:
        """Return writable repository slug and PR head owner."""
        self.ensure_authenticated()
        info = self.repo_info(upstream_slug)
        if info is None:
            raise ValueError(f"GitHub repository is not accessible: {upstream_slug}")
        permissions = info.get("permissions") or {}
        if permissions.get("push") is True:
            owner = (info.get("owner") or {}).get("login") or upstream_slug.split("/")[
                0
            ]
            return upstream_slug, str(owner)

        login = self.login()
        repo_name = upstream_slug.split("/", 1)[1]
        fork_slug = f"{login}/{repo_name}"
        fork = self.repo_info(fork_slug)
        if _is_fork_of(fork, upstream_slug):
            return fork_slug, login
        if fork is not None:
            raise RepositoryCollisionError(
                f"Contribution repository {fork_slug} contains unrelated content. "
                "Rename or remove it, then retry."
            )

        result = self._gh(
            ["repo", "fork", upstream_slug, "--clone=false"],
            check=False,
        )
        fork = self.repo_info(fork_slug)
        if _is_fork_of(fork, upstream_slug):
            return fork_slug, login
        if fork is not None:
            raise RepositoryCollisionError(
                f"Contribution repository {fork_slug} contains unrelated content. "
                "Rename or remove it, then retry."
            )
        detail = result.stderr.strip() or result.stdout.strip() or "not confirmed"
        raise ValueError(
            f"Could not create standard contributor fork {fork_slug}: {detail}. "
            "AnkiOps does not search for renamed forks. Nothing was pushed."
        )

    def find_pull_request(
        self, upstream_slug: str, head: str
    ) -> PullRequestInfo | None:
        result = self._gh(
            [
                "api",
                "-X",
                "GET",
                f"repos/{upstream_slug}/pulls",
                "-f",
                f"head={head}",
                "-f",
                "state=all",
                "-f",
                "per_page=1",
            ],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(f"Could not inspect pull requests: {detail}")
        try:
            values = json.loads(result.stdout)
            if not values:
                return None
            return _parse_pull_request(values[0])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("GitHub returned invalid pull request state.") from error

    def update_pr(self, url: str, *, title: str) -> None:
        result = self._gh(
            ["pr", "edit", url, "--title", title],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(f"GitHub did not update the pull request: {detail}")

    def create_pr(
        self,
        upstream_slug: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> str:
        result = self._gh(
            [
                "pr",
                "create",
                "--repo",
                upstream_slug,
                "--head",
                head,
                "--base",
                base,
                "--title",
                title,
                "--body",
                body,
            ],
            check=False,
        )
        url = result.stdout.strip()
        if result.returncode != 0 or not url:
            detail = result.stderr.strip() or url or "unknown error"
            raise ValueError(f"GitHub did not create the pull request: {detail}")
        return url


def _parse_pull_request(value: dict[str, Any]) -> PullRequestInfo:
    return PullRequestInfo(
        url=str(value["html_url"]),
        state="MERGED" if value.get("merged_at") else str(value["state"]).upper(),
        title=str(value["title"]),
        head_sha=str(value["head"]["sha"]),
    )

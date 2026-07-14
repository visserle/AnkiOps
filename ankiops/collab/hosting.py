"""Authenticated GitHub operations for collab repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ankiops.collab.errors import (
    RepositoryCollisionError,
    RepositoryCreationUncertainError,
)

MAX_FORK_NAME_ATTEMPTS = 20
MAX_FORK_DISCOVERY_RESULTS = 1000
MAX_GITHUB_REPOSITORY_NAME_LENGTH = 100


def _fork_name(repo_name: str, attempt: int) -> str:
    if attempt == 0:
        return repo_name
    suffix = "-ankiops" if attempt == 1 else f"-ankiops-{attempt}"
    base = repo_name[: MAX_GITHUB_REPOSITORY_NAME_LENGTH - len(suffix)]
    return f"{base}{suffix}"


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
    head_branch: str
    head_sha: str
    head_repository: str


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
            return None
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def create_repo(self, slug: str) -> None:
        self.ensure_authenticated()
        if self.repo_info(slug) is not None:
            raise RepositoryCollisionError(
                f"GitHub repository already exists: {slug}. Choose a new name."
            )
        result = self._gh(
            ["repo", "create", slug, "--public"],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RepositoryCreationUncertainError(
                f"Could not confirm creation of GitHub repository {slug}: {detail}"
            )

    def login(self) -> str:
        result = self._gh(["api", "user", "--jq", ".login"], check=False)
        login = result.stdout.strip()
        if result.returncode != 0 or not login:
            raise ValueError("GitHub CLI is not authenticated. Run: gh auth login")
        return login

    def _existing_fork(self, login: str, upstream_slug: str) -> str | None:
        result = self._gh(
            [
                "repo",
                "list",
                login,
                "--fork",
                "--limit",
                str(MAX_FORK_DISCOVERY_RESULTS + 1),
                "--json",
                "nameWithOwner,parent",
            ],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(f"Could not inspect existing GitHub forks: {detail}")
        try:
            repositories = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("GitHub returned an invalid fork list.") from error
        if not isinstance(repositories, list):
            raise ValueError("GitHub returned an invalid fork list.")
        if len(repositories) > MAX_FORK_DISCOVERY_RESULTS:
            raise ValueError(
                "Could not safely inspect all existing GitHub forks. "
                "Reduce the number of forks under this account, then retry."
            )
        for repository in repositories:
            if not isinstance(repository, dict):
                continue
            parent = repository.get("parent") or {}
            parent_slug = parent.get("nameWithOwner")
            fork_slug = repository.get("nameWithOwner")
            if (
                isinstance(parent_slug, str)
                and parent_slug.casefold() == upstream_slug.casefold()
                and isinstance(fork_slug, str)
                and fork_slug
            ):
                return fork_slug
        return None

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
        existing_fork = self._existing_fork(login, upstream_slug)
        if existing_fork is not None:
            return existing_fork, login
        repo_name = upstream_slug.split("/", 1)[1]
        for attempt in range(MAX_FORK_NAME_ATTEMPTS):
            fork_name = _fork_name(repo_name, attempt)
            fork_slug = f"{login}/{fork_name}"
            fork = self.repo_info(fork_slug)
            if fork is not None:
                if _is_fork_of(fork, upstream_slug):
                    return fork_slug, login
                continue

            args = ["repo", "fork", upstream_slug, "--clone=false"]
            if fork_name != repo_name:
                args.extend(["--fork-name", fork_name])
            result = self._gh(args, check=False)
            fork = self.repo_info(fork_slug)
            if _is_fork_of(fork, upstream_slug):
                return fork_slug, login
            if fork is not None:
                continue
            if result.returncode != 0:
                detail = (
                    result.stderr.strip() or result.stdout.strip() or "unknown error"
                )
                raise ValueError(
                    f"Could not create contributor fork {fork_slug}: {detail}. "
                    "Nothing was pushed; retrying is safe."
                )
            raise ValueError(
                f"GitHub did not confirm contributor fork {fork_slug}. "
                "Nothing was pushed; retrying is safe."
            )

        raise ValueError(
            f"Could not find an available fork name under {login}. "
            "Rename one of the colliding repositories, then retry."
        )

    def find_pull_request(
        self, upstream_slug: str, head: str
    ) -> PullRequestInfo | None:
        result = self._gh(
            [
                "pr",
                "list",
                "--repo",
                upstream_slug,
                "--head",
                head,
                "--state",
                "all",
                "--limit",
                "1",
                "--json",
                "url,state,headRefName,headRefOid,headRepository",
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
        url=str(value["url"]),
        state=str(value["state"]).upper(),
        head_branch=str(value["headRefName"]),
        head_sha=str(value["headRefOid"]),
        head_repository=str(value["headRepository"]["nameWithOwner"]),
    )

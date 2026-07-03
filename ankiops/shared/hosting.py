"""Authenticated GitHub operations for shared repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
            return
        result = self._gh(
            ["repo", "create", slug, "--public"],
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ValueError(f"Could not create GitHub repository {slug}: {detail}")

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
        if fork is None:
            result = self._gh(
                ["repo", "fork", upstream_slug, "--clone=false"], check=False
            )
            if result.returncode != 0:
                detail = (
                    result.stderr.strip() or result.stdout.strip() or "unknown error"
                )
                raise ValueError(
                    f"Could not create contributor fork {fork_slug}: {detail}. "
                    "Nothing was pushed; retrying is safe."
                )
            fork = self.repo_info(fork_slug)

        parent = (fork or {}).get("parent") or {}
        if not (fork or {}).get("fork") or parent.get("full_name") != upstream_slug:
            raise ValueError(
                f"Cannot use {fork_slug}: it is not a fork of {upstream_slug}."
            )
        return fork_slug, login

    def find_open_pr(self, upstream_slug: str, head: str) -> str | None:
        result = self._gh(
            [
                "pr",
                "list",
                "--repo",
                upstream_slug,
                "--head",
                head,
                "--state",
                "open",
                "--json",
                "url",
                "--jq",
                ".[0].url",
            ],
            check=False,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    def create_pr(
        self,
        upstream_slug: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> str:
        existing = self.find_open_pr(upstream_slug, head)
        if existing:
            return existing
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

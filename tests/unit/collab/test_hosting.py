from __future__ import annotations

import subprocess

import pytest

from ankiops.collab.errors import RepositoryCollisionError
from ankiops.collab.hosting import GitHubHost, PullRequestInfo


def test_github_repository_creation_is_always_public(tmp_path, monkeypatch):
    calls = []

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    GitHubHost(tmp_path).create_repo("owner/repo")

    assert calls == [(["repo", "create", "owner/repo", "--public"], False)]


def test_github_repository_creation_failure_is_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["gh"], 1, stdout="", stderr="connection closed after request"
        ),
    )

    with pytest.raises(ValueError, match="Retrying is safe"):
        GitHubHost(tmp_path).create_repo("owner/repo")


def test_missing_github_cli_gives_exact_setup_command(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.collab.hosting.shutil.which", lambda _name: None)

    with pytest.raises(ValueError, match="Install gh, then run: gh auth login"):
        GitHubHost(tmp_path).ensure_authenticated()


def test_invalid_github_auth_gives_exact_login_command(tmp_path, monkeypatch):
    failed = subprocess.CompletedProcess(
        ["gh", "auth", "status"],
        1,
        stdout="",
        stderr="token is invalid",
    )
    monkeypatch.setattr(GitHubHost, "_gh", lambda *_args, **_kwargs: failed)

    with pytest.raises(ValueError, match="Run: gh auth login") as error:
        GitHubHost(tmp_path).ensure_authenticated()

    assert "token is invalid" in str(error.value)


def test_repo_info_distinguishes_absent_from_unavailable(tmp_path, monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                ["gh"], 1, stdout="", stderr="gh: Not Found (HTTP 404)"
            ),
            subprocess.CompletedProcess(
                ["gh"], 1, stdout="", stderr="network unavailable"
            ),
        ]
    )
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: next(results),
    )
    host = GitHubHost(tmp_path)

    assert host.repo_info("owner/absent") is None
    with pytest.raises(ValueError, match="network unavailable"):
        host.repo_info("owner/unknown")


def _configure_contributor(monkeypatch, repositories):
    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )


def test_publish_target_uses_upstream_when_user_can_push(tmp_path, monkeypatch):
    repositories = {
        "owner/repo": {
            "permissions": {"push": True},
            "owner": {"login": "owner"},
        }
    }
    _configure_contributor(monkeypatch, repositories)
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: pytest.fail("a fork must not be created"),
    )

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "owner/repo",
        "owner",
    )


def test_publish_target_creates_deterministic_fork(tmp_path, monkeypatch):
    repositories = {"owner/repo": {"permissions": {"push": False}}}
    calls = []
    _configure_contributor(monkeypatch, repositories)

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        repositories["contributor/repo"] = {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        }
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo",
        "contributor",
    )
    assert calls == [
        (
            [
                "repo",
                "fork",
                "owner/repo",
                "--clone=false",
            ],
            False,
        )
    ]


def test_publish_target_reuses_deterministic_fork(tmp_path, monkeypatch):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        },
    }
    _configure_contributor(monkeypatch, repositories)
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: pytest.fail("the existing fork must be reused"),
    )

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo",
        "contributor",
    )


def test_publish_target_refuses_deterministic_name_collision(tmp_path, monkeypatch):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {"fork": False},
    }
    _configure_contributor(monkeypatch, repositories)

    with pytest.raises(RepositoryCollisionError, match="unrelated content"):
        GitHubHost(tmp_path).publish_target("owner/repo")


def test_publish_target_accepts_fork_after_ambiguous_creation(tmp_path, monkeypatch):
    repositories = {"owner/repo": {"permissions": {"push": False}}}
    _configure_contributor(monkeypatch, repositories)

    def ambiguous_gh(_host, args, *, check=True):
        repositories["contributor/repo"] = {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        }
        return subprocess.CompletedProcess(
            ["gh", *args],
            1,
            stdout="",
            stderr="connection closed after request",
        )

    monkeypatch.setattr(GitHubHost, "_gh", ambiguous_gh)

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo",
        "contributor",
    )


def test_publish_target_explains_differently_named_fork(tmp_path, monkeypatch):
    repositories = {"owner/repo": {"permissions": {"push": False}}}
    _configure_contributor(monkeypatch, repositories)
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["gh"], 1, stdout="", stderr="already forked"
        ),
    )

    with pytest.raises(ValueError, match="does not search for renamed forks"):
        GitHubHost(tmp_path).publish_target("owner/repo")


def test_pull_request_title_update_uses_existing_pr(tmp_path, monkeypatch):
    calls = []

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    GitHubHost(tmp_path).update_pr(
        "https://github.com/owner/repo/pull/7", title="Refine the answer"
    )

    assert calls == [
        (
            [
                "pr",
                "edit",
                "https://github.com/owner/repo/pull/7",
                "--title",
                "Refine the answer",
            ],
            False,
        )
    ]


def test_pull_request_lookup_filters_exact_fork_owner_and_branch(tmp_path, monkeypatch):
    calls = []

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(
            ["gh", *args],
            0,
            stdout=(
                '[{"html_url":"https://github.com/owner/repo/pull/7",'
                '"state":"open","title":"A title",'
                '"head":{"sha":"abc123"}}]'
            ),
            stderr="",
        )

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    assert GitHubHost(tmp_path).find_pull_request(
        "owner/repo", "contributor:ankiops/contribution"
    ) == PullRequestInfo(
        url="https://github.com/owner/repo/pull/7",
        state="OPEN",
        title="A title",
        head_sha="abc123",
    )
    assert calls == [
        (
            [
                "api",
                "-X",
                "GET",
                "repos/owner/repo/pulls",
                "-f",
                "head=contributor:ankiops/contribution",
                "-f",
                "state=all",
                "-f",
                "per_page=1",
            ],
            False,
        )
    ]


def test_pull_request_lookup_distinguishes_merged_from_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["gh"],
            0,
            stdout=(
                '[{"html_url":"https://github.com/owner/repo/pull/7",'
                '"state":"closed","merged_at":"2026-07-14T16:00:00Z",'
                '"title":"Merged title",'
                '"head":{"sha":"abc123"}}]'
            ),
            stderr="",
        ),
    )

    pull_request = GitHubHost(tmp_path).find_pull_request(
        "owner/repo", "contributor:ankiops/contribution"
    )

    assert pull_request is not None
    assert pull_request.state == "MERGED"

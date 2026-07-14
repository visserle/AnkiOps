from __future__ import annotations

import json
import subprocess

import pytest

from ankiops.collab.errors import RepositoryCreationUncertainError
from ankiops.collab.hosting import (
    MAX_FORK_DISCOVERY_RESULTS,
    MAX_FORK_NAME_ATTEMPTS,
    GitHubHost,
)


def test_github_repository_creation_is_always_public(tmp_path, monkeypatch):
    calls = []

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(
            ["gh", *args],
            1 if args[:1] == ["api"] else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    GitHubHost(tmp_path).create_repo("owner/repo")

    assert (["repo", "create", "owner/repo", "--public"], False) in calls


def test_github_repository_creation_rejects_an_existing_target(tmp_path, monkeypatch):
    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "repo_info", lambda *_args: {"private": False})

    with pytest.raises(ValueError, match="already exists.*Choose a new name"):
        GitHubHost(tmp_path).create_repo("owner/repo")


def test_github_repository_creation_reports_an_ambiguous_failed_command(
    tmp_path, monkeypatch
):
    def fake_gh(_host, args, *, check=True):
        return subprocess.CompletedProcess(
            ["gh", *args],
            1 if args[:1] in (["api"], ["repo"]) else 0,
            stdout="",
            stderr="connection closed after request",
        )

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    with pytest.raises(
        RepositoryCreationUncertainError, match="Could not confirm creation"
    ):
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


def test_publish_target_uses_an_alternate_fork_name_when_default_is_occupied(
    tmp_path, monkeypatch
):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {"fork": False},
    }
    calls = []

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        repositories["contributor/repo-ankiops"] = {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        }
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo-ankiops",
        "contributor",
    )
    assert calls == [
        (
            [
                "repo",
                "fork",
                "owner/repo",
                "--clone=false",
                "--fork-name",
                "repo-ankiops",
            ],
            False,
        )
    ]


def test_publish_target_reuses_an_existing_alternate_fork_on_retry(
    tmp_path, monkeypatch
):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {"fork": False},
        "contributor/repo-ankiops": {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        },
    }

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(
        GitHubHost,
        "_existing_fork",
        lambda *_args: "contributor/repo-ankiops",
    )
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: pytest.fail("retry must not create another fork"),
    )

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo-ankiops",
        "contributor",
    )


def test_publish_target_reuses_a_renamed_existing_fork_without_mutation(
    tmp_path, monkeypatch
):
    calls = []

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: (
            {"permissions": {"push": False}} if slug == "owner/repo" else None
        ),
    )

    def fake_gh(_host, args, *, check=True):
        calls.append((args, check))
        if args[:2] == ["repo", "list"]:
            return subprocess.CompletedProcess(
                ["gh", *args],
                0,
                stdout=(
                    '[{"nameWithOwner":"contributor/my-renamed-fork",'
                    '"parent":{"nameWithOwner":"owner/repo"}}]'
                ),
                stderr="",
            )
        pytest.fail("an existing fork must never be renamed or recreated")

    monkeypatch.setattr(GitHubHost, "_gh", fake_gh)

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/my-renamed-fork",
        "contributor",
    )
    assert len(calls) == 1
    assert calls[0][0][:2] == ["repo", "list"]


def test_publish_target_does_not_mutate_when_existing_fork_lookup_fails(
    tmp_path, monkeypatch
):
    calls = []

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: (
            {"permissions": {"push": False}} if slug == "owner/repo" else None
        ),
    )

    def failed_lookup(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(
            ["gh", *args], 1, stdout="", stderr="API unavailable"
        )

    monkeypatch.setattr(GitHubHost, "_gh", failed_lookup)

    with pytest.raises(ValueError, match="inspect existing GitHub forks") as error:
        GitHubHost(tmp_path).publish_target("owner/repo")

    assert "API unavailable" in str(error.value)
    assert len(calls) == 1
    assert calls[0][0][:2] == ["repo", "list"]


def test_publish_target_does_not_mutate_when_fork_discovery_exceeds_its_bound(
    tmp_path, monkeypatch
):
    calls = []
    repositories = [
        {
            "nameWithOwner": f"contributor/fork-{index}",
            "parent": {"nameWithOwner": f"other/repo-{index}"},
        }
        for index in range(MAX_FORK_DISCOVERY_RESULTS + 1)
    ]

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: (
            {"permissions": {"push": False}} if slug == "owner/repo" else None
        ),
    )

    def full_fork_inventory(_host, args, *, check=True):
        calls.append((args, check))
        return subprocess.CompletedProcess(
            ["gh", *args], 0, stdout=json.dumps(repositories), stderr=""
        )

    monkeypatch.setattr(GitHubHost, "_gh", full_fork_inventory)

    with pytest.raises(ValueError, match="safely inspect all existing GitHub forks"):
        GitHubHost(tmp_path).publish_target("owner/repo")

    assert len(calls) == 1
    assert calls[0][0][:2] == ["repo", "list"]


def test_publish_target_reconciles_an_ambiguous_fork_command(tmp_path, monkeypatch):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {"fork": False},
    }

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )

    def ambiguous_gh(_host, args, *, check=True):
        assert args[-2:] == ["--fork-name", "repo-ankiops"]
        repositories["contributor/repo-ankiops"] = {
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
        "contributor/repo-ankiops",
        "contributor",
    )


def test_publish_target_accepts_githubs_canonical_parent_casing(tmp_path, monkeypatch):
    repositories = {
        "Owner/Repo": {"permissions": {"push": False}},
        "contributor/Repo": {"fork": False},
    }

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )

    def create_fork(_host, args, *, check=True):
        repositories["contributor/Repo-ankiops"] = {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        }
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", create_fork)

    assert GitHubHost(tmp_path).publish_target("Owner/Repo") == (
        "contributor/Repo-ankiops",
        "contributor",
    )


def test_publish_target_advances_after_a_fork_name_race(tmp_path, monkeypatch):
    repositories = {
        "owner/repo": {"permissions": {"push": False}},
        "contributor/repo": {"fork": False},
    }
    attempts = []

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )

    def racing_gh(_host, args, *, check=True):
        fork_name = args[-1]
        attempts.append(fork_name)
        if len(attempts) == 1:
            repositories[f"contributor/{fork_name}"] = {"fork": False}
            return subprocess.CompletedProcess(
                ["gh", *args], 1, stdout="", stderr="name already exists"
            )
        repositories[f"contributor/{fork_name}"] = {
            "fork": True,
            "parent": {"full_name": "owner/repo"},
        }
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", racing_gh)

    assert GitHubHost(tmp_path).publish_target("owner/repo") == (
        "contributor/repo-ankiops-2",
        "contributor",
    )
    assert attempts == ["repo-ankiops", "repo-ankiops-2"]


def test_publish_target_stops_after_bounded_fork_name_collisions(tmp_path, monkeypatch):
    candidate_lookups = []

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)

    def occupied_repo(_host, slug):
        if slug == "owner/repo":
            return {"permissions": {"push": False}}
        candidate_lookups.append(slug)
        return {"fork": False}

    monkeypatch.setattr(GitHubHost, "repo_info", occupied_repo)
    monkeypatch.setattr(
        GitHubHost,
        "_gh",
        lambda *_args, **_kwargs: pytest.fail(
            "occupied candidates must not trigger repository creation"
        ),
    )

    with pytest.raises(ValueError, match="available fork name"):
        GitHubHost(tmp_path).publish_target("owner/repo")

    assert len(candidate_lookups) == MAX_FORK_NAME_ATTEMPTS
    assert candidate_lookups[-1] == "contributor/repo-ankiops-19"


def test_publish_target_keeps_alternate_fork_names_within_github_limit(
    tmp_path, monkeypatch
):
    repo_name = "r" * 100
    upstream_slug = f"owner/{repo_name}"
    repositories = {
        upstream_slug: {"permissions": {"push": False}},
        f"contributor/{repo_name}": {"fork": False},
    }

    monkeypatch.setattr(GitHubHost, "ensure_authenticated", lambda *_args: None)
    monkeypatch.setattr(GitHubHost, "login", lambda *_args: "contributor")
    monkeypatch.setattr(GitHubHost, "_existing_fork", lambda *_args: None)
    monkeypatch.setattr(
        GitHubHost,
        "repo_info",
        lambda _host, slug: repositories.get(slug),
    )

    def create_fork(_host, args, *, check=True):
        fork_name = args[-1]
        repositories[f"contributor/{fork_name}"] = {
            "fork": True,
            "parent": {"full_name": upstream_slug},
        }
        return subprocess.CompletedProcess(["gh", *args], 0, stdout="", stderr="")

    monkeypatch.setattr(GitHubHost, "_gh", create_fork)

    fork_slug, _login = GitHubHost(tmp_path).publish_target(upstream_slug)
    fork_name = fork_slug.split("/", 1)[1]

    assert len(fork_name) == 100
    assert fork_name.endswith("-ankiops")


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

from __future__ import annotations

import subprocess

import pytest

from ankiops.shared.hosting import GitHubHost


def test_missing_github_cli_gives_exact_setup_command(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.shared.hosting.shutil.which", lambda _name: None)

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

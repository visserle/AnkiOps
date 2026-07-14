from __future__ import annotations

import logging
import subprocess

import pytest

from ankiops.git import GitRepository, git_snapshot


def _init_git_repo(collection_root):
    subprocess.run(
        ["git", "init"], cwd=collection_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=collection_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=collection_root,
        check=True,
    )


def _commit_all(collection_root, message):
    subprocess.run(["git", "add", "."], cwd=collection_root, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=collection_root, check=True)


def _git_status(collection_root):
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=collection_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _git_head(collection_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=collection_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _head_subject(collection_root):
    result = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=collection_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _head_name_status(collection_root):
    result = subprocess.run(
        ["git", "show", "--name-status", "--format=", "HEAD"],
        cwd=collection_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_anonymous_clone_disables_credentials_and_prompts(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("ankiops.git.subprocess.run", fake_run)

    target = tmp_path / "public"
    GitRepository.clone("https://github.com/owner/repo.git", target, anonymous=True)

    args, kwargs = calls[0]
    assert args[:3] == ["git", "-c", "credential.helper="]
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_failed_git_command_exposes_stderr_to_callers_and_debug_log(
    tmp_path, monkeypatch, caplog
):
    def fail_run(args, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            args,
            output="",
            stderr="error: failed to push some refs (non-fast-forward)\n",
        )

    monkeypatch.setattr("ankiops.git.subprocess.run", fail_run)

    with caplog.at_level(logging.DEBUG, logger="ankiops.git"):
        with pytest.raises(subprocess.CalledProcessError) as raised:
            GitRepository(tmp_path).run(["push", "publish", "HEAD:main"])

    assert "non-fast-forward" in str(raised.value)
    assert raised.value.stderr == (
        "error: failed to push some refs (non-fast-forward)\n"
    )
    assert "non-fast-forward" in caplog.text


def test_failed_git_command_redacts_credentials_from_exception_and_debug_log(
    tmp_path, monkeypatch, caplog
):
    secret = "ghp_0123456789abcdefghijklmnopqrstuvwxyz"
    remote_url = f"https://oauth2:{secret}@github.com/owner/repo.git"

    def fail_run(args, **_kwargs):
        raise subprocess.CalledProcessError(
            128,
            args,
            output=f"access_token={secret}\n",
            stderr=(
                f"fatal: Authentication failed for '{remote_url}'\n"
                f"Authorization: Bearer {secret}\n"
            ),
        )

    monkeypatch.setattr("ankiops.git.subprocess.run", fail_run)

    with caplog.at_level(logging.DEBUG, logger="ankiops.git"):
        with pytest.raises(subprocess.CalledProcessError) as raised:
            GitRepository(tmp_path).run(["fetch", remote_url])

    error = raised.value
    exposed = "\n".join(
        (str(error), error.stdout or "", error.stderr or "", caplog.text)
    )
    assert secret not in exposed
    assert secret not in " ".join(error.cmd)
    assert "Authentication failed" in exposed
    assert "<redacted>" in exposed


def test_unchecked_git_result_raises_with_actionable_safe_detail(tmp_path, monkeypatch):
    secret = "github_pat_0123456789abcdefghijklmnopqrstuvwxyz"

    def failed_result(args, **_kwargs):
        return subprocess.CompletedProcess(
            args,
            128,
            stdout="",
            stderr=f"fatal: unable to access repository: token={secret}\n",
        )

    monkeypatch.setattr("ankiops.git.subprocess.run", failed_result)

    result = GitRepository(tmp_path).run(["ls-remote", "upstream"], check=False)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        result.check_returncode()

    assert "unable to access repository" in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in (raised.value.stderr or "")


def test_git_snapshot_commits_only_scoped_paths(tmp_path):
    _init_git_repo(tmp_path)
    scoped = tmp_path / "Deck.md"
    outside = tmp_path / "Other.md"
    scoped.write_text("old\n", encoding="utf-8")
    outside.write_text("old\n", encoding="utf-8")
    _commit_all(tmp_path, "root")

    scoped.write_text("new\n", encoding="utf-8")
    outside.write_text("new\n", encoding="utf-8")

    assert git_snapshot(tmp_path, action="test", paths=[scoped])

    assert _head_subject(tmp_path) == "AnkiOps: snapshot before test"
    show = _head_name_status(tmp_path)
    assert "M\tDeck.md" in show
    assert "Other.md" not in show
    assert _git_status(tmp_path) == " M Other.md\n"


def test_git_snapshot_ignores_scoped_empty_directories(tmp_path):
    _init_git_repo(tmp_path)
    baseline = tmp_path / ".gitignore"
    baseline.write_text(".ankiops.db\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    private_deck = tmp_path / "Private Draft.md"
    private_deck.write_text("private\n", encoding="utf-8")
    empty_media = tmp_path / "media"
    empty_media.mkdir()

    assert git_snapshot(
        tmp_path,
        action="files-to-anki",
        paths=[private_deck, empty_media],
    )

    assert _head_subject(tmp_path) == "AnkiOps: snapshot before files-to-anki"
    assert _head_name_status(tmp_path) == "A\tPrivate Draft.md\n"
    assert _git_status(tmp_path) == ""


def test_git_snapshot_commits_deleted_tracked_scoped_path(tmp_path):
    _init_git_repo(tmp_path)
    deck = tmp_path / "Deck.md"
    deck.write_text("old\n", encoding="utf-8")
    _commit_all(tmp_path, "root")

    deck.unlink()

    assert git_snapshot(tmp_path, action="delete test", paths=[deck])
    assert "D\tDeck.md" in _head_name_status(tmp_path)
    assert _git_status(tmp_path) == ""


def test_git_snapshot_skips_clean_scope(tmp_path):
    _init_git_repo(tmp_path)
    scoped = tmp_path / "Deck.md"
    outside = tmp_path / "Other.md"
    scoped.write_text("old\n", encoding="utf-8")
    outside.write_text("old\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    initial_head = _git_head(tmp_path)

    outside.write_text("new\n", encoding="utf-8")

    assert not git_snapshot(tmp_path, action="clean test", paths=[scoped])
    assert _git_head(tmp_path) == initial_head
    assert _git_status(tmp_path) == " M Other.md\n"


def test_git_snapshot_collection_path_keeps_broad_behavior(tmp_path):
    _init_git_repo(tmp_path)
    first = tmp_path / "Deck.md"
    second = tmp_path / "Other.md"
    first.write_text("old\n", encoding="utf-8")
    second.write_text("old\n", encoding="utf-8")
    _commit_all(tmp_path, "root")

    first.write_text("new\n", encoding="utf-8")
    second.write_text("new\n", encoding="utf-8")

    assert git_snapshot(tmp_path, action="broad test", paths=[tmp_path])

    show = _head_name_status(tmp_path)
    assert "M\tDeck.md" in show
    assert "M\tOther.md" in show
    assert _git_status(tmp_path) == ""


def test_status_lines_keep_unicode_paths_readable(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "Déck Ω — punctuation!.md"
    path.write_text("content\n", encoding="utf-8")

    assert GitRepository(tmp_path).status_lines() == ["?? Déck Ω — punctuation!.md"]


def test_diff_name_status_preserves_unicode_spaces_and_rename_paths(tmp_path):
    _init_git_repo(tmp_path)
    old_path = tmp_path / "Old déck Ω.md"
    old_path.write_text("content\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    subprocess.run(
        ["git", "mv", old_path.name, "New déck Ω.md"],
        cwd=tmp_path,
        check=True,
    )
    _commit_all(tmp_path, "rename")

    repository = GitRepository(tmp_path)
    changes = repository.diff_name_status("HEAD^", "HEAD")

    assert [(change.status, change.paths) for change in changes] == [
        ("R100", ("Old déck Ω.md", "New déck Ω.md"))
    ]


def test_worktree_matches_uses_final_non_ignored_files_without_changing_index(
    tmp_path,
):
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("Ignored.txt\n", encoding="utf-8")
    deck = tmp_path / "Deck.md"
    deck.write_text("original\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    repository = GitRepository(tmp_path)

    deck.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "Deck.md"], cwd=tmp_path, check=True)
    deck.write_text("original\n", encoding="utf-8")
    (tmp_path / "Ignored.txt").write_text("secret\n", encoding="utf-8")
    cached_before = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    assert repository.worktree_matches("HEAD")
    assert (
        subprocess.run(
            ["git", "diff", "--cached"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        == cached_before
    )

    deck.write_text("final worktree\n", encoding="utf-8")
    assert not repository.worktree_matches("HEAD")

    deck.write_text("original\n", encoding="utf-8")
    (tmp_path / "New.md").write_text("new\n", encoding="utf-8")
    assert not repository.worktree_matches("HEAD")


def test_worktree_matches_does_not_write_an_untracked_symlink_target(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("Private.txt\n", encoding="utf-8")
    (tmp_path / "Baseline.md").write_text("baseline\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    (tmp_path / "Private.txt").write_text("must stay private\n", encoding="utf-8")
    (tmp_path / "Shared.link").symlink_to("Private.txt")
    private_oid = subprocess.run(
        ["git", "hash-object", "Private.txt"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    assert not GitRepository(tmp_path).worktree_matches("HEAD")
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", private_oid],
            cwd=tmp_path,
            text=True,
            capture_output=True,
        ).returncode
        != 0
    )


def test_git_snapshot_propagates_checkpoint_failure(tmp_path, monkeypatch):
    _init_git_repo(tmp_path)
    tracked = tmp_path / "Deck.md"
    tracked.write_text("old\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    tracked.write_text("new\n", encoding="utf-8")

    def fail_commit(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["git", "commit"])

    monkeypatch.setattr(GitRepository, "run", fail_commit)

    with pytest.raises(ValueError, match="required Git checkpoint"):
        git_snapshot(tmp_path, action="test", paths=[tracked])


def test_delete_remote_branch_with_lease_refuses_an_advanced_branch(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(remote)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    local = tmp_path / "local"
    local.mkdir()
    _init_git_repo(local)
    subprocess.run(["git", "remote", "add", "publish", str(remote)], cwd=local)
    tracked = local / "Deck.md"
    tracked.write_text("first\n", encoding="utf-8")
    _commit_all(local, "first")
    expected = _git_head(local)
    subprocess.run(
        ["git", "push", "publish", "HEAD:refs/heads/ankiops/test"],
        cwd=local,
        check=True,
        capture_output=True,
    )
    tracked.write_text("advanced\n", encoding="utf-8")
    _commit_all(local, "advanced")
    advanced = _git_head(local)
    subprocess.run(
        ["git", "push", "publish", "HEAD:refs/heads/ankiops/test"],
        cwd=local,
        check=True,
        capture_output=True,
    )

    with pytest.raises(subprocess.CalledProcessError):
        GitRepository(local).delete_remote_branch_with_lease(
            "publish", "ankiops/test", expected
        )

    assert GitRepository(local).remote_branch_sha("publish", "ankiops/test") == advanced

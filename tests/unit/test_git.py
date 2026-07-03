from __future__ import annotations

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


def test_trees_equal_is_false_when_either_ref_is_missing(tmp_path):
    _init_git_repo(tmp_path)
    tracked = tmp_path / "Deck.md"
    tracked.write_text("content\n", encoding="utf-8")
    _commit_all(tmp_path, "root")
    repository = GitRepository(tmp_path)

    assert not repository.trees_equal("HEAD", "missing-ref")
    assert not repository.trees_equal("missing-ref", "also-missing")


def test_status_lines_keep_unicode_paths_readable(tmp_path):
    _init_git_repo(tmp_path)
    path = tmp_path / "Déck Ω — punctuation!.md"
    path.write_text("content\n", encoding="utf-8")

    assert GitRepository(tmp_path).status_lines() == [
        '?? "Déck Ω — punctuation!.md"'
    ]


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

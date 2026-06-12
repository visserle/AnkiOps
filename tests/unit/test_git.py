from __future__ import annotations

import subprocess

from ankiops.git import git_snapshot


def _init_git_repo(collection_dir):
    subprocess.run(["git", "init"], cwd=collection_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=collection_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=collection_dir,
        check=True,
    )


def _commit_all(collection_dir, message):
    subprocess.run(["git", "add", "."], cwd=collection_dir, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=collection_dir, check=True)


def _git_status(collection_dir):
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _git_head(collection_dir):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _head_subject(collection_dir):
    result = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _head_name_status(collection_dir):
    result = subprocess.run(
        ["git", "show", "--name-status", "--format=", "HEAD"],
        cwd=collection_dir,
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

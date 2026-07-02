"""Tests for strict collection/profile requirements."""

from __future__ import annotations

import subprocess

import pytest

from ankiops.collection import _setup_gitignore, require_collection_dir
from ankiops.sync.state import SyncState


def _init_git(path):
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True)


def test_require_collection_dir_exits_when_profile_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.collection.get_collection_dir", lambda: tmp_path)

    db = SyncState.open(tmp_path)
    db.close()
    _init_git(tmp_path)

    with pytest.raises(SystemExit):
        require_collection_dir("Default")


def test_require_collection_dir_exits_on_profile_mismatch(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr("ankiops.collection.get_collection_dir", lambda: tmp_path)

    db = SyncState.open(tmp_path)
    try:
        db.set_profile_name("Work")
    finally:
        db.close()
    _init_git(tmp_path)
    monkeypatch.setattr("sys.argv", ["ankiops", "fa"])

    with pytest.raises(SystemExit):
        require_collection_dir("Default")
    assert "Nothing was changed" in caplog.text
    assert "retry: ankiops fa" in caplog.text


def test_require_collection_dir_returns_path_on_profile_match(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.collection.get_collection_dir", lambda: tmp_path)

    db = SyncState.open(tmp_path)
    try:
        db.set_profile_name("Work")
    finally:
        db.close()

    _init_git(tmp_path)

    assert require_collection_dir("Work") == tmp_path


def test_require_collection_dir_exits_without_root_git(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.collection.get_collection_dir", lambda: tmp_path)
    db = SyncState.open(tmp_path)
    db.close()

    with pytest.raises(SystemExit):
        require_collection_dir()


def test_collection_gitignore_excludes_nested_repositories_and_recovery(tmp_path):
    _setup_gitignore(tmp_path)

    entries = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/shared/" in entries
    assert ".ankiops/" in entries

"""Tests for strict collection/profile requirements."""

from __future__ import annotations

import pytest

from ankiops.config import require_collection_dir
from ankiops.db import SQLiteDbAdapter


def test_require_collection_dir_exits_when_profile_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: tmp_path)

    db = SQLiteDbAdapter.open(tmp_path)
    db.close()

    with pytest.raises(SystemExit):
        require_collection_dir("Default")


def test_require_collection_dir_exits_on_profile_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: tmp_path)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        db.set_profile_name("Work")
    finally:
        db.close()

    with pytest.raises(SystemExit):
        require_collection_dir("Default")


def test_require_collection_dir_returns_path_on_profile_match(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: tmp_path)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        db.set_profile_name("Work")
    finally:
        db.close()

    assert require_collection_dir("Work") == tmp_path

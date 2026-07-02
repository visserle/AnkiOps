"""Invalid collection databases are rejected without compatibility behavior."""

from __future__ import annotations

import sqlite3

import pytest

from ankiops.collection import ANKIOPS_DB
from ankiops.sync.state import SyncState


def test_unreadable_database_is_rejected_without_mutation(tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    db_path.write_bytes(b"not a sqlite database")
    original = db_path.read_bytes()

    with pytest.raises(ValueError, match="does not migrate or recreate"):
        SyncState.open(tmp_path)

    assert db_path.read_bytes() == original
    assert not (tmp_path / f"{ANKIOPS_DB}.corrupt").exists()


def test_old_schema_is_rejected_without_mutation(tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE old_notes (note_key TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO old_notes VALUES ('preserve-me')")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="fresh collection"):
        SyncState.open(tmp_path)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("SELECT * FROM old_notes").fetchall() == [
            ("preserve-me",)
        ]
    finally:
        connection.close()


def test_existing_backup_file_is_not_touched(tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    backup = tmp_path / f"{ANKIOPS_DB}.corrupt"
    db_path.write_bytes(b"invalid")
    backup.write_text("user artifact", encoding="utf-8")

    with pytest.raises(ValueError):
        SyncState.open(tmp_path)

    assert db_path.read_bytes() == b"invalid"
    assert backup.read_text(encoding="utf-8") == "user artifact"

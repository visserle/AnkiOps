"""Invalid collection databases are backed up and recreated once."""

from __future__ import annotations

import sqlite3

import pytest

from ankiops.collection import ANKIOPS_DB
from ankiops.sync.state import SyncState


def test_unreadable_database_is_backed_up_and_recreated(tmp_path, caplog):
    db_path = tmp_path / ANKIOPS_DB
    db_path.write_bytes(b"not a sqlite database")
    original = db_path.read_bytes()

    state = SyncState.open(tmp_path)
    try:
        state.upsert_note_links([("new-key", 123)])
        assert state.resolve_note_ids(["new-key"]) == {"new-key": 123}
    finally:
        state.close()

    assert (tmp_path / f"{ANKIOPS_DB}.corrupt").read_bytes() == original
    assert "ankiops init" in caplog.text


def test_old_schema_is_backed_up_and_recreated(tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE old_notes (note_key TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO old_notes VALUES ('preserve-me')")
    connection.commit()
    connection.close()

    state = SyncState.open(tmp_path)
    state.close()

    connection = sqlite3.connect(tmp_path / f"{ANKIOPS_DB}.corrupt")
    try:
        assert connection.execute("SELECT * FROM old_notes").fetchall() == [
            ("preserve-me",)
        ]
    finally:
        connection.close()


def test_existing_backup_file_is_replaced_by_latest_corrupt_database(tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    backup = tmp_path / f"{ANKIOPS_DB}.corrupt"
    db_path.write_bytes(b"invalid")
    backup.write_text("user artifact", encoding="utf-8")

    state = SyncState.open(tmp_path)
    state.close()

    assert db_path.read_bytes() != b"invalid"
    assert backup.read_bytes() == b"invalid"


def test_second_database_creation_failure_is_propagated(tmp_path, monkeypatch):
    db_path = tmp_path / ANKIOPS_DB
    db_path.write_bytes(b"invalid")

    def fail_connect(*_args, **_kwargs):
        raise sqlite3.DatabaseError("still broken")

    monkeypatch.setattr("ankiops.sync.state.sqlite3.connect", fail_connect)

    with pytest.raises(ValueError, match="could not be recreated"):
        SyncState.open(tmp_path)

    assert (tmp_path / f"{ANKIOPS_DB}.corrupt").read_bytes() == b"invalid"

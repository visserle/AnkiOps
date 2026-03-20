"""Corrupted collection scenarios for DB recovery paths."""

from __future__ import annotations

import sqlite3

from ankiops.config import ANKIOPS_DB
from ankiops.db import SQLiteDbAdapter
from tests.support.assertions import assert_summary


def test_corr_db_001_unreadable_db_is_backed_up_and_recreated(world):
    """CORR-DB-001."""
    with world.db_session() as db:
        db.upsert_note_links([("corr-001", 1)])

    world.corrupt_db()

    with world.db_session() as recovered:
        assert recovered.resolve_note_ids(["corr-001"]).get("corr-001") is None

    assert (world.root / f"{ANKIOPS_DB}.corrupt").exists()


def test_corr_db_002_import_continues_after_db_recovery(world):
    """CORR-DB-002."""
    with world.db_session():
        pass
    world.corrupt_db()

    with world.db_session() as recovered:
        world.write_qa_deck("RecoveredImportDeck", [("Recover Q", "Recover A", None)])
        result = world.sync_import(recovered)

        assert_summary(result.summary, created=1, errors=0)
        assert len(world.mock_anki.notes) == 1


def test_corr_db_003_schema_corruption_is_auto_recovered(tmp_path):
    """CORR-DB-003."""
    db_path = tmp_path / ANKIOPS_DB

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE notes (note_key TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    recovered = SQLiteDbAdapter.open(tmp_path)
    try:
        recovered.upsert_note_links([("schema-key", 101)])
    finally:
        recovered.close()

    check_db = SQLiteDbAdapter.open(tmp_path)
    try:
        assert check_db.resolve_note_ids(["schema-key"]).get("schema-key") == 101
    finally:
        check_db.close()

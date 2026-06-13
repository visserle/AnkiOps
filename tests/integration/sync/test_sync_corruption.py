"""Corrupted collection scenarios for DB recovery paths."""

from __future__ import annotations

import sqlite3

from ankiops.collection import ANKIOPS_DB
from ankiops.sync.state import SyncState
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


def test_corr_db_003_import_rebuilds_mapping_from_embedded_key(world):
    """CORR-DB-003."""
    note_key = "corr-import-rebind"
    note_id = world.add_qa_note(
        deck_name="RecoveredImportDeck",
        question="Recover Q",
        answer="Old A",
        note_key=note_key,
    )
    world.write_qa_deck("RecoveredImportDeck", [("Recover Q", "New A", note_key)])

    with world.db_session() as db:
        db.upsert_note_links([(note_key, note_id)])

    world.corrupt_db()

    with world.db_session() as recovered:
        assert recovered.resolve_note_ids([note_key]).get(note_key) is None

        result = world.sync_import(recovered)

        assert_summary(
            result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert len(world.mock_anki.notes) == 1
        assert note_id in world.mock_anki.notes
        assert world.mock_anki.notes[note_id]["fields"]["Answer"]["value"] == "New A"
        assert recovered.resolve_note_ids([note_key]).get(note_key) == note_id

    assert (world.root / f"{ANKIOPS_DB}.corrupt").exists()


def test_corr_db_004_schema_corruption_is_auto_recovered(tmp_path):
    """CORR-DB-004."""
    db_path = tmp_path / ANKIOPS_DB

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE notes (note_key TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    recovered = SyncState.open(tmp_path)
    try:
        recovered.upsert_note_links([("schema-key", 101)])
    finally:
        recovered.close()

    check_db = SyncState.open(tmp_path)
    try:
        assert check_db.resolve_note_ids(["schema-key"]).get("schema-key") == 101
    finally:
        check_db.close()


def test_corr_db_005_existing_corrupt_backup_is_replaced(tmp_path):
    """CORR-DB-005."""
    db_path = tmp_path / ANKIOPS_DB
    corrupt_path = tmp_path / f"{ANKIOPS_DB}.corrupt"
    db_path.write_text("corrupt data", encoding="utf-8")
    corrupt_path.write_text("old corrupt backup", encoding="utf-8")

    recovered = SyncState.open(tmp_path)
    try:
        recovered.upsert_note_links([("recovered-key", 101)])
    finally:
        recovered.close()

    check_db = SyncState.open(tmp_path)
    try:
        assert check_db.resolve_note_ids(["recovered-key"])["recovered-key"] == 101
    finally:
        check_db.close()
    assert corrupt_path.read_text(encoding="utf-8") == "corrupt data"

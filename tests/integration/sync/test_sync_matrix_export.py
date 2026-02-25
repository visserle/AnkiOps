"""Export matrix scenarios (Anki -> Markdown) for full-sync behavior."""

from __future__ import annotations

from ankiops.config import ANKIOPS_DB
from ankiops.db import SQLiteDbAdapter
from tests.support.assertions import assert_summary


def _assert_deck_contains(world, deck_name: str, *parts: str) -> None:
    content = world.read_deck(deck_name)
    for part in parts:
        assert part in content


def test_exp_fresh_create_001_exports_anki_note_to_markdown(world):
    """EXP-FRESH-CREATE-001."""
    note_id = world.add_qa_note(
        deck_name="FreshExportDeck",
        question="Q1",
        answer="A1",
    )

    with world.db_session() as db:
        result = world.sync_export(db)

        assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)

        file_path = world.deck_path("FreshExportDeck")
        assert file_path.exists()
        _assert_deck_contains(world, "FreshExportDeck", "Q: Q1", "A: A1")

        keys = world.extract_note_keys("FreshExportDeck")
        assert len(keys) == 1
        assert db.get_note_key(note_id) == keys[0]


def test_exp_run_update_001_updates_existing_markdown_note(world):
    """EXP-RUN-UPDATE-001."""
    note_key = "exp-run-update-001"
    note_id = world.add_qa_note(
        deck_name="UpdateDeck",
        question="Q0",
        answer="A0",
        note_key=note_key,
    )

    world.write_qa_deck("UpdateDeck", [("Q0", "Old Markdown A", note_key)])
    world.set_note_answer(note_id, "New Anki A")

    with world.db_session() as db:
        db.set_note(note_key, note_id)

        result = world.sync_export(db)

        assert_summary(result.summary, created=0, updated=1, moved=0, deleted=0, errors=0)
        _assert_deck_contains(world, "UpdateDeck", "A: New Anki A")


def test_exp_run_delete_001_removes_deleted_anki_note_from_markdown_and_db(world):
    """EXP-RUN-DELETE-001."""
    keep_key = "exp-run-delete-keep"
    drop_key = "exp-run-delete-drop"

    keep_id = world.add_qa_note(
        deck_name="DeleteDeck",
        question="Keep Q",
        answer="Keep A",
        note_key=keep_key,
    )
    drop_id = world.add_qa_note(
        deck_name="DeleteDeck",
        question="Drop Q",
        answer="Drop A",
        note_key=drop_key,
    )

    world.write_qa_deck(
        "DeleteDeck",
        [
            ("Keep Q", "Keep A", keep_key),
            ("Drop Q", "Drop A", drop_key),
        ],
    )

    with world.db_session() as db:
        db.set_note(keep_key, keep_id)
        db.set_note(drop_key, drop_id)

        world.remove_note(drop_id)

        result = world.sync_export(db)

        assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)
        content = world.read_deck("DeleteDeck")
        assert "Keep Q" in content
        assert "Drop Q" not in content
        assert db.get_note_id(drop_key) is None


def test_exp_run_rename_001_renames_markdown_file_on_deck_rename(world):
    """EXP-RUN-RENAME-001."""
    note_key = "exp-run-rename-001"
    world.add_qa_note(
        deck_name="OldDeck",
        question="Deck Rename Q",
        answer="Deck Rename A",
        note_key=note_key,
    )

    with world.db_session() as db:
        world.sync_export(db)

        old_file = world.deck_path("OldDeck")
        assert old_file.exists()

        world.rename_deck("OldDeck", "NewDeck")

        world.sync_export(db)

        new_file = world.deck_path("NewDeck")
        assert new_file.exists()
        assert not old_file.exists()
        assert db.get_deck_id("NewDeck") is not None


def test_exp_run_drift_001_reuses_embedded_key_when_db_mapping_missing(world):
    """EXP-RUN-DRIFT-001."""
    note_key = "exp-run-drift-001"
    note_id = world.add_qa_note(
        deck_name="DriftDeck",
        question="Q Drift",
        answer="A Drift",
        note_key=note_key,
    )

    with world.db_session() as db:
        result = world.sync_export(db)

        assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
        _assert_deck_contains(world, "DriftDeck", f"<!-- note_key: {note_key} -->")
        assert db.get_note_key(note_id) == note_key


def test_exp_corr_drift_001_recovers_from_corrupt_db_and_rebuilds_mapping(world):
    """EXP-CORR-DRIFT-001."""
    note_key = "exp-corr-drift-001"
    note_id = world.add_qa_note(
        deck_name="CorruptDeck",
        question="Q Corrupt",
        answer="A Corrupt",
        note_key=note_key,
    )

    with world.db_session() as db:
        db.set_note("placeholder", 123)

    world.corrupt_db()

    with world.db_session() as recovered_db:
        result = world.sync_export(recovered_db)

        assert_summary(result.summary, created=1, errors=0)
        assert (world.root / f"{ANKIOPS_DB}.corrupt").exists()

    with world.db_session() as check_db:
        assert check_db.get_note_id(note_key) == note_id


def test_exp_run_delete_002_always_removes_orphan_markdown_files(world):
    """EXP-RUN-DELETE-002."""
    orphan_file = world.write_qa_deck(
        "OrphanDeck",
        [("Old Q", "Old A", "orphan-key")],
    )
    assert orphan_file.exists()

    with world.db_session() as db:
        result = world.sync_export(db)

    assert not orphan_file.exists()
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)


def test_exp_run_drift_003_stale_key_mapping_rebinds_to_embedded_key(world):
    """EXP-RUN-DRIFT-003."""
    note_id = world.add_qa_note(
        deck_name="StaleKeyDeck",
        question="Stale Key Q",
        answer="Stale Key A",
        note_key="embedded-good-key",
    )

    db = SQLiteDbAdapter.load(world.root)
    try:
        db.set_note("stale-wrong-key", note_id)

        result = world.sync_export(db)

        assert_summary(result.summary, errors=0)
        assert db.get_note_key(note_id) == "embedded-good-key"
    finally:
        db.close()

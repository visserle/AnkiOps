"""Import matrix scenarios (Markdown -> Anki) for full-sync behavior."""

from __future__ import annotations

import pytest

from tests.support.assertions import assert_summary


def test_imp_fresh_create_001_creates_note_and_writes_key(world):
    """IMP-FRESH-CREATE-001."""
    world.write_qa_deck("FreshDeck", [("Fresh Q", "Fresh A", None)])

    with world.db_session() as db:
        result = world.sync_import(db)

        assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
        assert len(world.mock_anki.notes) == 1

        note_keys = world.extract_note_keys("FreshDeck")
        assert len(note_keys) == 1

        note_id = next(iter(world.mock_anki.notes.keys()))
        assert (
            world.mock_anki.notes[note_id]["fields"]["AnkiOps Key"]["value"]
            == note_keys[0]
        )
        assert db.get_note_id(note_keys[0]) == note_id


def test_imp_run_update_001_updates_existing_note(world):
    """IMP-RUN-UPDATE-001."""
    note_key = "imp-run-update-001"
    note_id = world.add_qa_note(
        deck_name="RunDeck",
        question="Original Q",
        answer="Original A",
        note_key=note_key,
    )

    world.write_qa_deck("RunDeck", [("Original Q", "Updated A", note_key)])

    with world.db_session() as db:
        db.set_note(note_key, note_id)

        result = world.sync_import(db)

        assert_summary(result.summary, created=0, updated=1, moved=0, deleted=0, errors=0)
        assert world.mock_anki.notes[note_id]["fields"]["Answer"]["value"] == "Updated A"


def test_imp_run_move_001_moves_note_between_decks(world):
    """IMP-RUN-MOVE-001."""
    note_key = "imp-run-move-001"
    note_id = world.add_qa_note(
        deck_name="SourceDeck",
        question="Move Q",
        answer="Move A",
        note_key=note_key,
    )

    world.write_qa_deck("SourceDeck", [])
    world.write_qa_deck("TargetDeck", [("Move Q", "Move A", note_key)])

    with world.db_session() as db:
        db.set_note(note_key, note_id)

        result = world.sync_import(db)

        assert_summary(result.summary, created=0, updated=0, moved=1, deleted=0, errors=0)
        card_id = world.mock_anki.notes[note_id]["cards"][0]
        assert world.mock_anki.cards[card_id]["deckName"] == "TargetDeck"


def test_imp_run_delete_001_deletes_orphaned_anki_note(world):
    """IMP-RUN-DELETE-001."""
    note_key = "imp-run-delete-001"
    note_id = world.add_qa_note(
        deck_name="DeleteDeck",
        question="Delete Q",
        answer="Delete A",
        note_key=note_key,
    )

    world.write_qa_deck("DeleteDeck", [])

    with world.db_session() as db:
        db.set_note(note_key, note_id)

        result = world.sync_import(db)

        assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)
        assert note_id not in world.mock_anki.notes


def test_imp_run_drift_001_recovers_missing_mapping_from_embedded_key(world):
    """IMP-RUN-DRIFT-001."""
    note_key = "imp-run-drift-001"
    note_id = world.add_qa_note(
        deck_name="DriftDeck",
        question="Drift Q",
        answer="Drift A",
        note_key=note_key,
    )

    world.write_qa_deck("DriftDeck", [("Drift Q", "Drift A", note_key)])

    with world.db_session() as db:
        result = world.sync_import(db)

        assert_summary(result.summary, created=0, deleted=0, errors=0)
        assert len(world.mock_anki.notes) == 1
        assert db.get_note_id(note_key) == note_id


def test_imp_run_drift_002_stale_mapping_rebinds_without_duplicate(world):
    """IMP-RUN-DRIFT-002."""
    note_key = "imp-run-drift-002"
    real_note_id = world.add_qa_note(
        deck_name="DriftDeck",
        question="Drift2 Q",
        answer="Drift2 A",
        note_key=note_key,
    )

    world.write_qa_deck("DriftDeck", [("Drift2 Q", "Drift2 A", note_key)])

    with world.db_session() as db:
        db.set_note(note_key, 999999)
        result = world.sync_import(db)

        assert_summary(result.summary, errors=0)
        assert len(world.mock_anki.notes) == 1
        assert db.get_note_id(note_key) == real_note_id


def test_imp_run_conflict_001_duplicate_note_keys_fail_fast(world):
    """IMP-RUN-CONFLICT-001."""
    duplicate_note_key = "duplicate-key"

    world.write_qa_deck("DeckA", [("A", "A1", duplicate_note_key)])
    world.write_qa_deck("DeckB", [("B", "B1", duplicate_note_key)])

    with world.db_session() as db:
        with pytest.raises(ValueError, match="Duplicate note_key"):
            world.sync_import(db)


def test_imp_run_update_002_note_type_mismatch_records_error_and_skips_update(world):
    """IMP-RUN-UPDATE-002."""
    note_key = "imp-run-type-mismatch-001"
    world.mock_anki.add_note(
        "MismatchDeck",
        "AnkiOpsCloze",
        {"Text": "{{c1::x}}", "AnkiOps Key": note_key},
    )
    note_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("MismatchDeck", [("Q mismatch", "A mismatch", note_key)])

    with world.db_session() as db:
        db.set_note(note_key, note_id)
        result = world.sync_import(db)

        assert len(result.results) == 1
        sync_res = result.results[0]
        assert len(sync_res.errors) == 1
        assert "Note type mismatch" in sync_res.errors[0]
        assert_summary(sync_res.summary, created=0, updated=0, moved=0, deleted=0)
        assert world.mock_anki.notes[note_id]["modelName"] == "AnkiOpsCloze"


def test_imp_run_drift_003_reports_untracked_anki_decks(world):
    """IMP-RUN-DRIFT-003."""
    world.mock_anki.add_note(
        "AnkiOnlyDeck",
        "AnkiOpsQA",
        {"Question": "Only in Anki", "Answer": "A", "AnkiOps Key": "anki-only-key"},
    )
    note_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        result = world.sync_import(db)

        assert result.results == []
        assert len(result.untracked_decks) == 1
        untracked = result.untracked_decks[0]
        assert untracked.deck_name == "AnkiOnlyDeck"
        assert note_id in untracked.note_ids

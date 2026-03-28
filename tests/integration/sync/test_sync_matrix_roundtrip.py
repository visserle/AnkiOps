"""Round-trip matrix scenarios."""

from __future__ import annotations

import pytest

from ankiops.config import ANKIOPS_DB
from ankiops.fingerprints import note_fingerprint
from tests.support.assertions import assert_summary


def test_rt_fresh_create_001_import_export_import_is_idempotent(world):
    """RT-FRESH-CREATE-001."""
    world.write_qa_deck("RoundTripFresh", [("RT Q1", "RT A1", None)])

    with world.db_session() as db:
        first_import = world.sync_import(db)
        assert_summary(
            first_import.summary, created=1, updated=0, moved=0, deleted=0, errors=0
        )

        export_result = world.sync_export(db)
        assert_summary(
            export_result.summary, created=0, updated=0, moved=0, deleted=0, errors=0
        )

        second_import = world.sync_import(db)
        assert_summary(
            second_import.summary, created=0, updated=0, moved=0, deleted=0, errors=0
        )

        keys = world.extract_note_keys("RoundTripFresh")
        assert len(keys) == 1
        assert len(world.mock_anki.notes) == 1


def test_rt_run_update_001_anki_change_then_export_then_import_noops(world):
    """RT-RUN-UPDATE-001."""
    world.write_qa_deck("RoundTripRun", [("Run Q", "Run A", None)])

    with world.db_session() as db:
        world.sync_import(db)

        note_id = next(iter(world.mock_anki.notes.keys()))
        world.set_note_answer(note_id, "Run A Modified In Anki")

        export_result = world.sync_export(db)
        assert_summary(
            export_result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert "A: Run A Modified In Anki" in world.read_deck("RoundTripRun")

        import_again = world.sync_import(db)
        assert_summary(
            import_again.summary, created=0, updated=0, moved=0, deleted=0, errors=0
        )
        assert (
            world.mock_anki.notes[note_id]["fields"]["Answer"]["value"]
            == "Run A Modified In Anki"
        )


def test_rt_run_update_002_anki_literal_highlight_roundtrip_rehydrates_mark_html(world):
    """RT-RUN-UPDATE-002."""
    world.write_qa_deck("RoundTripHighlight", [("Run Q", "Run A", None)])

    with world.db_session() as db:
        world.sync_import(db)

        note_id = next(iter(world.mock_anki.notes.keys()))
        world.mock_anki.notes[note_id]["fields"]["Answer"] = {
            "value": "<div>Run A with ==highlight==</div>"
        }

        export_result = world.sync_export(db)
        assert_summary(
            export_result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert "A: Run A with ==highlight==" in world.read_deck("RoundTripHighlight")

        import_result = world.sync_import(db)
        assert_summary(
            import_result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert (
            "<mark>highlight</mark>"
            in world.mock_anki.notes[note_id]["fields"]["Answer"]["value"]
        )

        export_again = world.sync_export(db)
        assert_summary(
            export_again.summary, created=0, updated=0, moved=0, deleted=0, errors=0
        )
        assert "A: Run A with ==highlight==" in world.read_deck("RoundTripHighlight")


def test_rt_run_update_003_directional_cache_isolation(world):
    """RT-RUN-UPDATE-003."""
    world.write_qa_deck("RoundTripCacheIso", [("Iso Q", "Iso A0", None)])

    with world.db_session() as db:
        world.sync_import(db)
        note_id = next(iter(world.mock_anki.notes.keys()))
        note = world.mock_anki.notes[note_id]
        note_key = note["fields"]["AnkiOps Key"]["value"]
        deck_path = world.deck_path("RoundTripCacheIso")

        # Export should ignore import cache and use only export cache.
        world.set_note_answer(note_id, "Iso A1")
        local_md_note = world.fs.read_markdown_file(deck_path).notes[0]
        local_md_hash = note_fingerprint(local_md_note.note_type, local_md_note.fields)
        observed_anki_fields = {
            name: field_data["value"]
            for name, field_data in world.mock_anki.notes[note_id]["fields"].items()
        }
        observed_anki_hash = note_fingerprint(note["modelName"], observed_anki_fields)
        db.upsert_import_hashes([(note_key, local_md_hash, observed_anki_hash)])
        db.upsert_export_hashes([(note_key, "stale-md", "stale-anki")])

        export_result = world.sync_export(db)
        assert_summary(
            export_result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert "A: Iso A1" in world.read_deck("RoundTripCacheIso")

        # Import should ignore export cache and use only import cache.
        updated_md = world.read_deck("RoundTripCacheIso").replace("Iso A1", "Iso A2")
        deck_path.write_text(updated_md, encoding="utf-8")
        edited_md_note = world.fs.read_markdown_file(deck_path).notes[0]
        edited_md_hash = note_fingerprint(edited_md_note.note_type, edited_md_note.fields)
        current_anki_fields = {
            name: field_data["value"]
            for name, field_data in world.mock_anki.notes[note_id]["fields"].items()
        }
        current_anki_hash = note_fingerprint(note["modelName"], current_anki_fields)
        db.upsert_import_hashes([(note_key, "stale-md", "stale-anki")])
        db.upsert_export_hashes([(note_key, edited_md_hash, current_anki_hash)])

        import_result = world.sync_import(db)
        assert_summary(
            import_result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        assert world.mock_anki.notes[note_id]["fields"]["Answer"]["value"] == "Iso A2"


def test_rt_corr_drift_001_corrupt_db_between_cycles_recovers_cleanly(world):
    """RT-CORR-DRIFT-001."""
    world.write_qa_deck("RoundTripCorrupt", [("Corrupt Q", "Corrupt A", None)])

    with world.db_session() as db:
        first_import = world.sync_import(db)
        assert_summary(first_import.summary, created=1, errors=0)

    world.corrupt_db()

    with world.db_session() as recovered_db:
        export_result = world.sync_export(recovered_db)
        assert_summary(export_result.summary, errors=0)

        import_result = world.sync_import(recovered_db)
        assert_summary(import_result.summary, errors=0)

    assert len(world.mock_anki.notes) == 1
    assert (world.root / f"{ANKIOPS_DB}.corrupt").exists()


def test_rt_run_delete_001_roundtrip_preserves_delete(world):
    """RT-RUN-DELETE-001."""
    world.write_qa_deck("RoundTripDelete", [("Delete Q", "Delete A", None)])

    with world.db_session() as db:
        world.sync_import(db)
        note_id = next(iter(world.mock_anki.notes.keys()))

        world.write_qa_deck("RoundTripDelete", [])

        imp = world.sync_import(db)
        assert_summary(imp.summary, deleted=1, errors=0)
        assert note_id not in world.mock_anki.notes

        exp = world.sync_export(db)
        assert exp.summary.deleted >= 1
        assert exp.summary.errors == 0
        assert not world.deck_path("RoundTripDelete").exists()


def test_rt_run_orphan_001_mixed_orphan_export_does_not_resurrect_keyed_note(world):
    """RT-RUN-ORPHAN-001."""
    world.write_qa_deck(
        "RoundTripMixedOrphan",
        [
            ("Stale orphan Q", "Stale orphan A", "stale-orphan-key"),
            ("Draft orphan Q", "Draft orphan A", None),
        ],
    )

    with world.db_session() as db:
        export_result = world.sync_export(db)
        assert_summary(
            export_result.summary, created=0, updated=0, moved=0, deleted=1, errors=0
        )
        assert "Stale orphan Q" not in world.read_deck("RoundTripMixedOrphan")
        assert "Draft orphan Q" in world.read_deck("RoundTripMixedOrphan")

        import_result = world.sync_import(db)
        assert_summary(
            import_result.summary, created=1, updated=0, moved=0, deleted=0, errors=0
        )

        assert len(world.mock_anki.notes) == 1
        created_note = next(iter(world.mock_anki.notes.values()))
        created_key = created_note["fields"]["AnkiOps Key"]["value"]
        assert created_key != "stale-orphan-key"
        assert "stale-orphan-key" not in world.read_deck("RoundTripMixedOrphan")


def test_rt_run_move_001_roundtrip_preserves_move(world):
    """RT-RUN-MOVE-001."""
    world.write_qa_deck("RoundTripMoveA", [("Move Q", "Move A", None)])

    with world.db_session() as db:
        world.sync_import(db)
        note_id = next(iter(world.mock_anki.notes.keys()))

        content = world.read_deck("RoundTripMoveA")
        world.deck_path("RoundTripMoveA").write_text("", encoding="utf-8")
        world.deck_path("RoundTripMoveB").write_text(content, encoding="utf-8")

        imp = world.sync_import(db)
        assert_summary(imp.summary, created=0, updated=0, moved=1, deleted=0, errors=0)

        exp = world.sync_export(db)
        assert_summary(exp.summary, created=0, errors=0)

        card_id = world.mock_anki.notes[note_id]["cards"][0]
        assert world.mock_anki.cards[card_id]["deckName"] == "RoundTripMoveB"


def test_rt_fresh_update_001_export_import_export_keeps_markdown_winner(world):
    """RT-FRESH-UPDATE-001."""
    world.mock_anki.add_note(
        "RoundTripFreshUpdate",
        "AnkiOpsQA",
        {
            "Question": "Fresh UQ",
            "Answer": "From Anki",
            "AnkiOps Key": "rt-fresh-update-key",
        },
    )

    with world.db_session() as db:
        first_export = world.sync_export(db)
        assert_summary(first_export.summary, created=1, errors=0)

        md_file = world.deck_path("RoundTripFreshUpdate")
        content = md_file.read_text(encoding="utf-8").replace(
            "From Anki", "From Markdown"
        )
        md_file.write_text(content, encoding="utf-8")

        imp = world.sync_import(db)
        assert_summary(imp.summary, updated=1, errors=0)

        second_export = world.sync_export(db)
        assert_summary(second_export.summary, updated=0, errors=0)
        assert "A: From Markdown" in md_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "operation",
    ["update", "move"],
    ids=["RT-CORR-UPDATE-001", "RT-CORR-MOVE-001"],
)
def test_rt_corr_operations_recover_after_db_corruption(operation, world):
    world.write_qa_deck("RoundTripCorrA", [("Corr Q", "Corr A", None)])

    with world.db_session() as db:
        world.sync_import(db)

    if operation == "update":
        md_file = world.deck_path("RoundTripCorrA")
        md_file.write_text(
            md_file.read_text(encoding="utf-8").replace("Corr A", "Corr A2"),
            encoding="utf-8",
        )
    else:
        content = world.read_deck("RoundTripCorrA")
        world.deck_path("RoundTripCorrA").write_text("", encoding="utf-8")
        world.deck_path("RoundTripCorrB").write_text(content, encoding="utf-8")

    world.corrupt_db()

    with world.db_session() as recovered:
        imp = world.sync_import(recovered)

        if operation == "update":
            assert_summary(
                imp.summary, created=0, updated=1, moved=0, deleted=0, errors=0
            )
        else:
            assert_summary(
                imp.summary, created=0, updated=0, moved=1, deleted=0, errors=0
            )

        exp = world.sync_export(recovered)
        assert_summary(exp.summary, errors=0)


def test_rt_run_conflict_001_last_sync_direction_wins(world):
    """RT-RUN-CONFLICT-001."""
    world.write_qa_deck("RoundTripConflict", [("CQ", "Base", None)])

    with world.db_session() as db:
        world.sync_import(db)

        note_id = next(iter(world.mock_anki.notes.keys()))
        md_file = world.deck_path("RoundTripConflict")

        # Diverge both sides.
        md_file.write_text(
            md_file.read_text(encoding="utf-8").replace("Base", "Markdown Edit"),
            encoding="utf-8",
        )
        world.mock_anki.notes[note_id]["fields"]["Answer"] = {"value": "Anki Edit"}

        # Export makes markdown match Anki (export winner).
        world.sync_export(db)
        assert "A: Anki Edit" in md_file.read_text(encoding="utf-8")

        # Edit markdown again and import; now markdown wins.
        md_file.write_text(
            md_file.read_text(encoding="utf-8").replace("Anki Edit", "Markdown Final"),
            encoding="utf-8",
        )
        world.sync_import(db)
        assert (
            world.mock_anki.notes[note_id]["fields"]["Answer"]["value"]
            == "Markdown Final"
        )

"""Export matrix scenarios (Anki -> Markdown) for full-sync behavior."""

from __future__ import annotations

import logging

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

        assert_summary(
            result.summary, created=1, updated=0, moved=0, deleted=0, errors=0
        )

        file_path = world.deck_path("FreshExportDeck")
        assert file_path.exists()
        _assert_deck_contains(world, "FreshExportDeck", "Q: Q1", "A: A1")

        keys = world.extract_note_keys("FreshExportDeck")
        assert len(keys) == 1
        assert db.resolve_note_keys([note_id]).get(note_id) == keys[0]


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
        db.upsert_note_links([(note_key, note_id)])

        result = world.sync_export(db)

        assert_summary(
            result.summary, created=0, updated=1, moved=0, deleted=0, errors=0
        )
        _assert_deck_contains(world, "UpdateDeck", "A: New Anki A")


def test_exp_fresh_create_002_preserves_blockquote_citation_links(world):
    """EXP-FRESH-CREATE-002."""
    world.add_qa_note(
        deck_name="CitationDeck",
        question="Q Citation",
        answer=(
            '<blockquote><p>called. <a href="https://en.wikipedia.org/wiki/Pygmalion'
            '#cite_note-13">[13]</a><br>aus Pygmalion von George Bernard Shaw'
            "</p></blockquote>"
        ),
    )

    with world.db_session() as db:
        result = world.sync_export(db)

    assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
    _assert_deck_contains(
        world,
        "CitationDeck",
        "Q: Q Citation",
        "A: > called. [[13]](<https://en.wikipedia.org/wiki/Pygmalion#cite_note-13>)",
        "> aus Pygmalion von George Bernard Shaw",
    )


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
        db.upsert_note_links([(keep_key, keep_id)])
        db.upsert_note_links([(drop_key, drop_id)])

        world.remove_note(drop_id)

        result = world.sync_export(db)

        assert_summary(
            result.summary, created=0, updated=0, moved=0, deleted=1, errors=0
        )
        content = world.read_deck("DeleteDeck")
        assert "Keep Q" in content
        assert "Drop Q" not in content
        assert db.resolve_note_ids([drop_key]).get(drop_key) is None


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
        assert db.resolve_deck_id("NewDeck") is not None


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

        assert_summary(
            result.summary, created=1, updated=0, moved=0, deleted=0, errors=0
        )
        _assert_deck_contains(world, "DriftDeck", f"<!-- note_key: {note_key} -->")
        assert db.resolve_note_keys([note_id]).get(note_id) == note_key


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
        db.upsert_note_links([("placeholder", 123)])

    world.corrupt_db()

    with world.db_session() as recovered_db:
        result = world.sync_export(recovered_db)

        assert_summary(result.summary, created=1, errors=0)
        assert (world.root / f"{ANKIOPS_DB}.corrupt").exists()

    with world.db_session() as check_db:
        assert check_db.resolve_note_ids([note_key]).get(note_key) == note_id


def test_exp_run_delete_002_always_removes_orphan_markdown_files(world):
    """EXP-RUN-DELETE-002."""
    orphan_file = world.write_qa_deck(
        "OrphanDeck",
        [("Old Q", "Old A", "orphan-key")],
    )
    assert orphan_file.exists()

    with world.db_session() as db:
        db.upsert_note_links([("orphan-key", 999001)])
        result = world.sync_export(db)
        assert db.resolve_note_ids(["orphan-key"]).get("orphan-key") is None

    assert not orphan_file.exists()
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)
    assert result.protected_note_groups == []


def test_exp_run_protect_001_keeps_keyless_note_in_active_deck(world):
    """EXP-RUN-PROTECT-001."""
    keep_key = "exp-protect-active-keep"
    keep_id = world.add_qa_note(
        deck_name="ProtectDeck",
        question="Keep Q",
        answer="Keep A",
        note_key=keep_key,
    )
    world.write_qa_deck(
        "ProtectDeck",
        [
            ("Keep Q", "Keep A", keep_key),
            ("Draft Q", "Draft A", None),
        ],
    )

    with world.db_session() as db:
        db.upsert_note_links([(keep_key, keep_id)])
        result = world.sync_export(db)

    content = world.read_deck("ProtectDeck")
    assert "Keep Q" in content
    assert "Draft Q" in content
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=0, errors=0)
    assert len(result.protected_note_groups) == 1
    assert result.protected_note_groups[0].deck_name == "ProtectDeck"
    assert result.protected_note_groups[0].note_count == 1


def test_exp_run_protect_002_preserves_keyless_orphan_markdown_file(world):
    """EXP-RUN-PROTECT-002."""
    orphan_file = world.write_qa_deck(
        "ProtectedOrphanDeck",
        [("Draft orphan Q", "Draft orphan A", None)],
    )
    assert orphan_file.exists()

    with world.db_session() as db:
        result = world.sync_export(db)

    assert orphan_file.exists()
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=0, errors=0)
    assert len(result.protected_note_groups) == 1
    assert result.protected_note_groups[0].deck_name == "ProtectedOrphanDeck"
    assert result.protected_note_groups[0].note_count == 1


def test_exp_run_protect_005_prunes_keyed_notes_from_mixed_orphan_file(world):
    """EXP-RUN-PROTECT-005."""
    orphan_file = world.write_qa_deck(
        "MixedOrphanDeck",
        [
            ("Stale orphan Q", "Stale orphan A", "stale-orphan-key"),
            ("Draft orphan Q", "Draft orphan A", None),
        ],
    )
    assert orphan_file.exists()

    with world.db_session() as db:
        db.upsert_note_links([("stale-orphan-key", 999002)])
        result = world.sync_export(db)
        assert db.resolve_note_ids(["stale-orphan-key"]).get("stale-orphan-key") is None

    content = world.read_deck("MixedOrphanDeck")
    assert orphan_file.exists()
    assert "Draft orphan Q" in content
    assert "Stale orphan Q" not in content
    assert "<!-- note_key: stale-orphan-key -->" not in content
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)
    assert len(result.protected_note_groups) == 1
    assert result.protected_note_groups[0].deck_name == "MixedOrphanDeck"
    assert result.protected_note_groups[0].note_count == 1


def test_exp_run_protect_006_keeps_link_for_active_key_seen_in_orphan_file(world):
    """EXP-RUN-PROTECT-006."""
    active_key = "active-key"
    note_id = world.add_qa_note(
        deck_name="ActiveDeck",
        question="Active Q",
        answer="Active A",
        note_key=active_key,
    )
    world.write_qa_deck("ActiveDeck", [("Active Q", "Active A", active_key)])
    world.write_qa_deck("DuplicateOrphanDeck", [("Stale Q", "Stale A", active_key)])

    with world.db_session() as db:
        db.upsert_note_links([(active_key, note_id)])
        result = world.sync_export(db)
        assert db.resolve_note_ids([active_key]).get(active_key) == note_id

    assert not world.deck_path("DuplicateOrphanDeck").exists()
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)


def test_exp_run_protect_003_preserves_malformed_orphan_file(world, caplog):
    """EXP-RUN-PROTECT-003."""
    malformed_file = world.deck_path("MalformedOrphanDeck")
    malformed_file.write_text("UNKNOWN: bad\n", encoding="utf-8")

    with (
        caplog.at_level(logging.WARNING),
        world.db_session() as db,
    ):
        result = world.sync_export(db)

    assert malformed_file.exists()
    assert "could not be parsed during export cleanup" in caplog.text
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=0, errors=0)
    assert result.protected_note_groups == []


def test_exp_run_protect_004_keyless_match_does_not_duplicate_note(world):
    """EXP-RUN-PROTECT-004."""
    world.add_qa_note(
        deck_name="NoDuplicateDeck",
        question="Shared Q",
        answer="Shared A",
        note_key=None,
    )
    world.write_qa_deck(
        "NoDuplicateDeck",
        [("Shared Q", "Shared A", None)],
    )

    with world.db_session() as db:
        result = world.sync_export(db)

    content = world.read_deck("NoDuplicateDeck")
    assert content.count("Q: Shared Q") == 1
    assert len(world.extract_note_keys("NoDuplicateDeck")) == 1
    assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
    assert all(
        deck.deck_name != "NoDuplicateDeck" for deck in result.protected_note_groups
    )


def test_exp_run_drift_003_stale_key_mapping_rebinds_to_embedded_key(world):
    """EXP-RUN-DRIFT-003."""
    note_id = world.add_qa_note(
        deck_name="StaleKeyDeck",
        question="Stale Key Q",
        answer="Stale Key A",
        note_key="embedded-good-key",
    )

    db = SQLiteDbAdapter.open(world.root)
    try:
        db.upsert_note_links([("stale-wrong-key", note_id)])

        result = world.sync_export(db)

        assert_summary(result.summary, errors=0)
        assert db.resolve_note_keys([note_id]).get(note_id) == "embedded-good-key"
    finally:
        db.close()

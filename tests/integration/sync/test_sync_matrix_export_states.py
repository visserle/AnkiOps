"""Additional export matrix scenarios across collection states."""

from __future__ import annotations

import pytest

from ankiops.db import SQLiteDbAdapter

from tests.support.assertions import assert_summary
from tests.support.scenario_case import mk_state_cases

CREATE_CASES = mk_state_cases(
    fresh="EXP-FRESH-CREATE-002",
    run="EXP-RUN-CREATE-001",
    corr="EXP-CORR-CREATE-001",
)

UPDATE_CASES = mk_state_cases(
    fresh="EXP-FRESH-UPDATE-001",
    run="EXP-RUN-UPDATE-002",
    corr="EXP-CORR-UPDATE-001",
)

DELETE_CASES = mk_state_cases(
    fresh="EXP-FRESH-DELETE-001",
    run="EXP-RUN-DELETE-003",
    corr="EXP-CORR-DELETE-001",
)

RENAME_CASES = mk_state_cases(
    fresh="EXP-FRESH-RENAME-001",
    run="EXP-RUN-RENAME-002",
    corr="EXP-CORR-RENAME-001",
)

CONFLICT_CASES = mk_state_cases(
    fresh="EXP-FRESH-CONFLICT-001",
    run="EXP-RUN-CONFLICT-001",
    corr="EXP-CORR-CONFLICT-001",
)

DRIFT_CASES = mk_state_cases(
    fresh="EXP-FRESH-DRIFT-001",
    run="EXP-RUN-DRIFT-002",
    corr="EXP-CORR-DRIFT-002",
)


@pytest.mark.parametrize("case", CREATE_CASES, ids=lambda state_case: state_case.id)
def test_export_create_is_full_sync_across_states(case, world):
    world.add_qa_note(
        deck_name="ExportCreateDeck",
        question=f"Create Q {case.state}",
        answer=f"Create A {case.state}",
        note_key=f"exp-create-{case.state.lower()}",
    )

    with world.db_session(case.state) as db:
        result = world.sync_export(db)

    assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
    assert world.deck_path("ExportCreateDeck").exists()


@pytest.mark.parametrize("case", UPDATE_CASES, ids=lambda state_case: state_case.id)
def test_export_update_is_full_sync_across_states(case, world):
    note_key = f"exp-state-update-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="ExportUpdateDeck",
        question="Update Q",
        answer="Anki Answer",
        note_key=note_key,
    )

    world.write_qa_deck("ExportUpdateDeck", [("Update Q", "Markdown Old", note_key)])

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        result = world.sync_export(db)

    assert_summary(result.summary, created=0, updated=1, moved=0, deleted=0, errors=0)
    assert "A: Anki Answer" in world.read_deck("ExportUpdateDeck")


@pytest.mark.parametrize("case", DELETE_CASES, ids=lambda state_case: state_case.id)
def test_export_delete_removes_orphan_file_across_states(case, world):
    orphan_file = world.write_qa_deck(
        "ExportDeleteDeck",
        [("Old Q", "Old A", "old-key")],
    )

    with world.db_session(case.state) as db:
        result = world.sync_export(db)

    assert not orphan_file.exists()
    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)


@pytest.mark.parametrize("case", RENAME_CASES, ids=lambda state_case: state_case.id)
def test_export_deck_rename_converges_to_new_file_across_states(case, world):
    note_key = f"exp-state-rename-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="NewDeckName",
        question="Rename Export Q",
        answer="Rename Export A",
        note_key=note_key,
    )

    old_file = world.write_qa_deck(
        "OldDeckName",
        [("Rename Export Q", "Rename Export A", note_key)],
    )

    deck_map = {}
    if case.state in {"RUN", "CORR"}:
        deck_map = {"OldDeckName": world.mock_anki.decks["NewDeckName"]}

    with world.db_session(
        case.state, note_map={note_key: note_id}, deck_map=deck_map
    ) as db:
        world.sync_export(db)

    assert world.deck_path("NewDeckName").exists()
    assert not old_file.exists()


@pytest.mark.parametrize("case", CONFLICT_CASES, ids=lambda state_case: state_case.id)
def test_export_conflict_prefers_anki_content_across_states(case, world):
    note_key = f"exp-state-conflict-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="ConflictExportDeck",
        question="Conflict Q",
        answer="Anki Winner",
        note_key=note_key,
    )

    world.write_qa_deck(
        "ConflictExportDeck", [("Conflict Q", "Markdown Loser", note_key)]
    )

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        world.sync_export(db)

    assert "A: Anki Winner" in world.read_deck("ConflictExportDeck")


@pytest.mark.parametrize("case", DRIFT_CASES, ids=lambda state_case: state_case.id)
def test_export_sets_missing_deck_mapping_across_states(case, world):
    world.add_qa_note(
        deck_name="DeckMappingDeck",
        question="Deck Q",
        answer="Deck A",
        note_key=f"deck-map-{case.state.lower()}",
    )
    deck_id = world.mock_anki.decks["DeckMappingDeck"]

    with world.db_session(case.state) as db:
        world.sync_export(db)
        assert db.get_deck_id("DeckMappingDeck") == deck_id


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

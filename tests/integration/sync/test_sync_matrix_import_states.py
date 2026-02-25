"""Additional import matrix scenarios across collection states."""

from __future__ import annotations

import pytest

from tests.support.assertions import assert_summary
from tests.support.scenario_case import mk_state_cases

CREATE_CASES = mk_state_cases(
    fresh="IMP-FRESH-CREATE-002",
    run="IMP-RUN-CREATE-001",
    corr="IMP-CORR-CREATE-001",
)

UPDATE_CASES = mk_state_cases(
    fresh="IMP-FRESH-UPDATE-001",
    run="IMP-RUN-UPDATE-003",
    corr="IMP-CORR-UPDATE-001",
)

DELETE_CASES = mk_state_cases(
    fresh="IMP-FRESH-DELETE-001",
    run="IMP-RUN-DELETE-002",
    corr="IMP-CORR-DELETE-001",
)

MOVE_CASES = mk_state_cases(
    fresh="IMP-FRESH-MOVE-001",
    run="IMP-RUN-MOVE-002",
    corr="IMP-CORR-MOVE-001",
)

CONFLICT_CASES = mk_state_cases(
    fresh="IMP-FRESH-CONFLICT-001",
    run="IMP-RUN-CONFLICT-002",
    corr="IMP-CORR-CONFLICT-001",
)

RENAME_CASES = mk_state_cases(
    fresh="IMP-FRESH-RENAME-001",
    run="IMP-RUN-RENAME-001",
    corr="IMP-CORR-RENAME-001",
)


@pytest.mark.parametrize("case", CREATE_CASES, ids=lambda state_case: state_case.id)
def test_import_create_is_full_sync_across_states(case, world):
    world.write_qa_deck(
        f"CreateDeck{case.state}",
        [(f"Create Q {case.state}", f"Create A {case.state}", None)],
    )

    with world.db_session(case.state) as db:
        result = world.sync_import(db)

    assert_summary(result.summary, created=1, updated=0, moved=0, deleted=0, errors=0)
    assert len(world.mock_anki.notes) == 1


@pytest.mark.parametrize("case", UPDATE_CASES, ids=lambda state_case: state_case.id)
def test_import_update_is_full_sync_across_states(case, world):
    note_key = f"imp-state-update-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="UpdateStateDeck",
        question="State Q",
        answer="Old A",
        note_key=note_key,
    )

    world.write_qa_deck("UpdateStateDeck", [("State Q", "New A", note_key)])

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        result = world.sync_import(db)

    assert_summary(result.summary, created=0, updated=1, moved=0, deleted=0, errors=0)
    assert world.mock_anki.notes[note_id]["fields"]["Answer"]["value"] == "New A"


@pytest.mark.parametrize("case", DELETE_CASES, ids=lambda state_case: state_case.id)
def test_import_delete_is_full_sync_across_states(case, world):
    note_key = f"imp-state-delete-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="DeleteStateDeck",
        question="Delete State Q",
        answer="Delete State A",
        note_key=note_key,
    )

    world.write_qa_deck("DeleteStateDeck", [])

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        result = world.sync_import(db)

    assert_summary(result.summary, created=0, updated=0, moved=0, deleted=1, errors=0)
    assert note_id not in world.mock_anki.notes


@pytest.mark.parametrize("case", MOVE_CASES, ids=lambda state_case: state_case.id)
def test_import_move_is_full_sync_across_states(case, world):
    note_key = f"imp-state-move-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="SourceStateDeck",
        question="Move State Q",
        answer="Move State A",
        note_key=note_key,
    )

    world.write_qa_deck("SourceStateDeck", [])
    world.write_qa_deck("TargetStateDeck", [("Move State Q", "Move State A", note_key)])

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        result = world.sync_import(db)

    assert_summary(result.summary, created=0, updated=0, moved=1, deleted=0, errors=0)
    card_id = world.mock_anki.notes[note_id]["cards"][0]
    assert world.mock_anki.cards[card_id]["deckName"] == "TargetStateDeck"

    assert len(world.mock_anki.notes) == 1


@pytest.mark.parametrize("case", CONFLICT_CASES, ids=lambda state_case: state_case.id)
def test_import_duplicate_key_conflict_fails_across_states(case, world):
    duplicate_note_key = f"dup-{case.state.lower()}"

    world.write_qa_deck("ConflictA", [("A", "A1", duplicate_note_key)])
    world.write_qa_deck("ConflictB", [("B", "B1", duplicate_note_key)])

    with world.db_session(case.state) as db:
        with pytest.raises(ValueError, match="Duplicate note_key"):
            world.sync_import(db)


@pytest.mark.parametrize("case", RENAME_CASES, ids=lambda state_case: state_case.id)
def test_import_rename_subdeck_is_full_sync_across_states(case, world):
    note_key = f"imp-state-rename-{case.state.lower()}"
    note_id = world.add_qa_note(
        deck_name="OldDeck::Sub",
        question="Rename Q",
        answer="Rename A",
        note_key=note_key,
    )

    world.write_qa_deck("OldDeck::Sub", [])
    world.write_qa_deck("NewDeck::Sub", [("Rename Q", "Rename A", note_key)])

    with world.db_session(case.state, note_map={note_key: note_id}) as db:
        world.sync_import(db)

    card_id = world.mock_anki.notes[note_id]["cards"][0]
    assert world.mock_anki.cards[card_id]["deckName"] == "NewDeck::Sub"

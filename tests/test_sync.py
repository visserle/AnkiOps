from pathlib import Path

import pytest

from ankiops.anki_to_markdown import _reconcile_blocks
from ankiops.models import Change, ChangeType, FileState, Note

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def sample_notes():
    return [
        Note(
            note_id=1, note_type="AnkiOpsQA", fields={"Question": "Q1", "Answer": "A1"}
        ),
        Note(
            note_id=None,
            note_type="AnkiOpsQA",
            fields={"Question": "Q2", "Answer": "A2"},
        ),
        Note(
            note_id=2, note_type="AnkiOpsQA", fields={"Question": "Q3", "Answer": "A3"}
        ),
    ]


@pytest.fixture
def file_state(sample_notes):
    return FileState(
        file_path=Path("dummy.md"),
        raw_content="",
        deck_id=101,
        parsed_notes=sample_notes,
    )


# -- Tests: FileState properties ---------------------------------------------


def test_file_state_existing_notes(file_state):
    """FileState.existing_notes should return only notes with IDs."""
    existing = file_state.existing_notes
    assert len(existing) == 2
    assert {n.note_id for n in existing} == {1, 2}


def test_file_state_new_notes(file_state):
    """FileState.new_notes should return only notes without IDs."""
    new = file_state.new_notes
    assert len(new) == 1
    assert new[0].fields["Question"] == "Q2"


# -- Tests: _reconcile_blocks ------------------------------------------------


def test_reconcile_blocks_no_existing():
    """First export: no existing file, should just sort by creation date (note_id)."""
    block_by_id = {
        "note_id: 2": (2, "Block 2"),
        "note_id: 1": (1, "Block 1"),
    }
    changes = _reconcile_blocks(block_by_id, existing_blocks={})

    assert len(changes) == 2
    assert changes[0].change_type == ChangeType.CREATE
    assert changes[0].entity_id == 1
    assert changes[0].entity_repr == "note_id: 1"

    assert changes[1].change_type == ChangeType.CREATE
    assert changes[1].entity_id == 2


def test_reconcile_blocks_with_updates_and_deletes():
    """Existing file: preserve order, update content, detect deletes."""
    existing = {
        "note_id: 1": "Block 1 (old)",  # Updated
        "note_id: 2": "Block 2",  # Skipped (no change)
        "note_id: 3": "Block 3",  # Deleted (not in block_by_id)
    }
    block_by_id = {
        "note_id: 1": (1, "Block 1 (new)"),
        "note_id: 2": (2, "Block 2"),
        "note_id: 4": (4, "Block 4"),  # Created
    }

    changes = _reconcile_blocks(block_by_id, existing_blocks=existing)

    # Expected order of processing in _reconcile_blocks:
    # 1. Existing blocks (1, 2, 3)
    # 2. New blocks (4)

    # Change 1: Update note 1
    assert changes[0].change_type == ChangeType.UPDATE
    assert changes[0].entity_id == 1
    assert changes[0].context["block_content"] == "Block 1 (new)"

    # Change 2: Skip note 2
    assert changes[1].change_type == ChangeType.SKIP
    assert changes[1].entity_id == 2

    # Change 3: Delete note 3
    assert changes[2].change_type == ChangeType.DELETE
    assert changes[2].entity_repr == "note_id: 3"
    assert changes[2].entity_id == 3  # Now parsed from block_id

    # Change 4: Create note 4
    assert changes[3].change_type == ChangeType.CREATE
    assert changes[3].entity_id == 4

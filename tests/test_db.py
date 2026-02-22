"""Tests for the Key Mapping system and recovery logic.

Includes:
1.  Basic CRUD operations (SQLite backed).
2.  Recovery from missing mappings using AnkiOps Key lookup.
3.  Corruption handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ankiops.config import ANKIOPS_DB
from ankiops.db import AnkiOpsDB
from ankiops.markdown_to_anki import _build_anki_actions, _sync_file
from ankiops.models import AnkiState, ChangeType, FileState, Note

# -- Basic CRUD Tests --------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Create a temporary AnkiOpsDB for testing."""
    return AnkiOpsDB.load(tmp_path)


def test_load_creates_tables(db, tmp_path):
    """Test that loading creates the necessary tables."""
    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()


def test_load_empty(db):
    """Loading from a non-existent file should return empty map."""
    assert db.get_note_id("any-key") is None
    assert db.get_deck_id("any-key") is None
    assert db.get_config("any-config") is None
    db.close()


def test_persistence(tmp_path):
    """Test that mappings and config persist across loads."""
    # First instance
    db1 = AnkiOpsDB.load(tmp_path)
    db1.set_deck("deck-1", 100)
    db1.set_note("note-1", 200)
    db1.set_config("profile", "default")
    db1.close()

    # Reload
    db2 = AnkiOpsDB.load(tmp_path)
    assert db2.get_deck_id("deck-1") == 100
    assert db2.get_note_id("note-1") == 200
    assert db2.get_config("profile") == "default"
    db2.close()


def test_config_mapping(db):
    """Test setting and getting config values."""
    db.set_config("test-key", "test-value")
    assert db.get_config("test-key") == "test-value"

    # Overwrite
    db.set_config("test-key", "new-value")
    assert db.get_config("test-key") == "new-value"
    db.close()


def test_note_mapping(db):
    """Test setting and getting note mappings."""
    key = "test-note-key"
    note_id = 67890

    db.set_note(key, note_id)
    assert db.get_note_id(key) == note_id
    assert db.get_note_key(note_id) == key

    # Non-existent
    assert db.get_note_id("key-2") is None
    db.close()


def test_deck_mapping(db):
    """Test setting and getting deck mappings."""
    key = "test-deck-key"
    deck_id = 12345

    db.set_deck(key, deck_id)
    assert db.get_deck_id(key) == deck_id
    assert db.get_deck_key(deck_id) == key

    assert db.get_note_id("deck-key-A") is None  # Should not be in notes
    db.close()


def test_overwrite_mapping(db):
    """Test updating an existing mapping enforces 1:1 behavior where possible."""
    db.set_note("key-1", 101)

    # Reassign key-1 to new note ID
    db.set_note("key-1", 999)

    assert db.get_note_id("key-1") == 999
    # The old ID 101 is mostly orphaned in the reverse sense
    assert db.get_note_key(101) is None

    db.close()


def test_generate_key():
    """Test key generation."""
    key = AnkiOpsDB.generate_key()
    assert isinstance(key, str)
    assert len(key) > 10  # Reasonable length check for UUID-like string


# -- Recovery and Sync Logic Tests -------------------------------------------


@pytest.fixture
def mock_db_instance(tmp_path):
    # Using real DB or mock? The original test mocked it, but real is safer.
    # Let's mock to strictly control return values for missing IDs.
    return MagicMock(spec=AnkiOpsDB)


@pytest.fixture
def mock_invoke():
    with patch("ankiops.markdown_to_anki.invoke") as mock:
        yield mock


@pytest.fixture
def mock_db_class():
    with patch("ankiops.markdown_to_anki.AnkiOpsDB") as mock:
        mock.generate_key.return_value = "generated-key"
        yield mock


@pytest.fixture
def mock_converter():
    return MagicMock()


@pytest.fixture
def anki_state():
    return AnkiState(
        deck_ids_by_name={"Deck1": 1},
        deck_names_by_id={1: "Deck1"},
        notes_by_id={},
        cards_by_id={},
        note_ids_by_deck_name={},
    )


def test_recovery_success(
    mock_db_instance, mock_invoke, anki_state, mock_converter
):
    """Test that a missing ID map entry is recovered via AnkiOps Key lookup."""

    # 1. Setup Markdown Note with a Key
    note_key = "key-1234"
    note = Note(
        note_key=note_key,
        note_type="AnkiOpsQA",
        fields={"Question": "Q", "Answer": "A"},
    )
    file_state = FileState(
        file_path=Path("test.md"),
        raw_content="",
        deck_key="deck-key-1",
        parsed_notes=[note],
    )

    # 2. Setup ID Map to return None (missing)
    mock_db_instance.get_note_id.return_value = None
    mock_db_instance.get_deck_id.return_value = 1  # Resolve deck
    mock_db_instance.get_deck_key.return_value = "deck-key-1"

    # 3. Setup Anki to find the note via "AnkiOps Key"
    recovered_note_id = 999

    def side_effect(action, **kwargs):
        if action == "findNotes":
            return [recovered_note_id]
        if action == "notesInfo":
            return [
                {
                    "noteId": recovered_note_id,
                    "modelName": "AnkiOpsQA",
                    "fields": {
                        "Question": {"value": "Q"},
                        "Answer": {"value": "A"},
                        "AnkiOps Key": {"value": note_key},
                    },
                }
            ]
        return []

    mock_invoke.side_effect = side_effect

    # 4. Run Sync
    result, _ = _sync_file(
        file_state, anki_state, mock_converter, mock_db_instance
    )

    # Should have called findNotes
    mock_invoke.assert_any_call("findNotes", query=f'"AnkiOps Key:{note_key}"')

    # Should have updated ID Map
    mock_db_instance.set_note.assert_called_with(note_key, recovered_note_id)


def test_create_injects_id(
    mock_db_instance,
    mock_invoke,
    anki_state,
    mock_converter,
    mock_db_class,
):
    """Test that creating a note injects the AnkiOps Key field."""
    note = Note(
        note_key=None, note_type="AnkiOpsQA", fields={"Question": "Q", "Answer": "A"}
    )

    file_state = FileState(
        file_path=Path("test.md"),
        raw_content="",
        deck_key="deck-key-1",
        parsed_notes=[note],
    )

    mock_db_instance.get_note_id.return_value = None
    mock_db_instance.get_deck_id.return_value = 1
    mock_db_instance.get_deck_key.return_value = "deck-key-1"

    # Run Sync
    result, _ = _sync_file(
        file_state, anki_state, mock_converter, mock_db_instance
    )

    # Check changes
    change = result.changes[0]
    assert change.change_type == ChangeType.CREATE
    # Generated key from AnkiOpsDB.generate_key()
    assert change.context["note_key"] == "generated-key"

    # Now verify _build_anki_actions injects it
    actions, _, _, _, _ = _build_anki_actions(
        "Deck1", False, result.changes, anki_state, result
    )

    # Check addNote params
    add_action = actions[0]
    assert add_action["action"] == "addNote"
    fields = add_action["params"]["note"]["fields"]
    assert fields["AnkiOps Key"] == "generated-key"


def test_corruption_recovery(tmp_path):
    """Test that if the DB is corrupt, it attempts to recover."""
    # Create a valid DB with one entry
    db = AnkiOpsDB.load(tmp_path)
    db.set_note("n1", 100)
    db.close()

    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()

    # Corrupt the DB file
    with open(db_path, "w") as f:
        f.write("corrupt data")

    # Attempt to load
    # It should detect corruption (SQLiteError), backup .db -> .db.corrupt,
    # and start fresh
    db = AnkiOpsDB.load(tmp_path)

    # Should be empty (fresh DB)
    assert db.get_note_id("n1") is None
    # Backup should exist
    assert (tmp_path / (ANKIOPS_DB + ".corrupt")).exists()
    db.close()

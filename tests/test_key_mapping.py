"""Tests for the Key Mapping system and recovery logic.

Includes:
1.  Basic CRUD operations (SQLite backed).
2.  Recovery from missing mappings using AnkiOps Key lookup.
3.  Corruption handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ankiops.config import KEY_MAP_DB
from ankiops.key_map import KeyMap
from ankiops.markdown_to_anki import _build_anki_actions, _sync_file
from ankiops.models import AnkiState, ChangeType, FileState, Note

# -- Basic CRUD Tests --------------------------------------------------------


@pytest.fixture
def key_map(tmp_path):
    """Create a temporary KeyMap for testing."""
    return KeyMap.load(tmp_path)


def test_load_creates_tables(key_map, tmp_path):
    """Test that loading creates the necessary tables."""
    db_path = tmp_path / KEY_MAP_DB
    assert db_path.exists()


def test_load_empty(key_map):
    """Loading from a non-existent file should return empty map."""
    assert key_map.get_note_id("any-key") is None
    assert key_map.get_deck_id("any-key") is None
    key_map.close()


def test_persistence(tmp_path):
    """Test that mappings persist across loads."""
    # First instance
    km1 = KeyMap.load(tmp_path)
    km1.set_deck("deck-1", 100)
    km1.set_note("note-1", 200)
    km1.close()

    # Reload
    km2 = KeyMap.load(tmp_path)
    assert km2.get_deck_id("deck-1") == 100
    assert km2.get_note_id("note-1") == 200
    km2.close()


def test_note_mapping(key_map):
    """Test setting and getting note mappings."""
    key = "test-note-key"
    note_id = 67890

    key_map.set_note(key, note_id)
    assert key_map.get_note_id(key) == note_id
    assert key_map.get_note_key(note_id) == key

    # Non-existent
    assert key_map.get_note_id("key-2") is None
    key_map.close()


def test_deck_mapping(key_map):
    """Test setting and getting deck mappings."""
    key = "test-deck-key"
    deck_id = 12345

    key_map.set_deck(key, deck_id)
    assert key_map.get_deck_id(key) == deck_id
    assert key_map.get_deck_key(deck_id) == key

    assert key_map.get_note_id("deck-key-A") is None  # Should not be in notes
    key_map.close()


def test_overwrite_mapping(key_map):
    """Test updating an existing mapping enforces 1:1 behavior where possible."""
    key_map.set_note("key-1", 101)

    # Reassign key-1 to new note ID
    key_map.set_note("key-1", 999)

    assert key_map.get_note_id("key-1") == 999
    # The old ID 101 is mostly orphaned in the reverse sense
    assert key_map.get_note_key(101) is None

    key_map.close()


def test_generate_key():
    """Test key generation."""
    key = KeyMap.generate_key()
    assert isinstance(key, str)
    assert len(key) > 10  # Reasonable length check for UUID-like string


# -- Recovery and Sync Logic Tests -------------------------------------------


@pytest.fixture
def mock_key_map_instance(tmp_path):
    # Using real DB or mock? The original test mocked it, but real is safer.
    # Let's mock to strictly control return values for missing IDs.
    return MagicMock(spec=KeyMap)


@pytest.fixture
def mock_invoke():
    with patch("ankiops.markdown_to_anki.invoke") as mock:
        yield mock


@pytest.fixture
def mock_key_map_class():
    with patch("ankiops.markdown_to_anki.KeyMap") as mock:
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
    mock_key_map_instance, mock_invoke, anki_state, mock_converter
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
    mock_key_map_instance.get_note_id.return_value = None
    mock_key_map_instance.get_deck_id.return_value = 1  # Resolve deck
    mock_key_map_instance.get_deck_key.return_value = "deck-key-1"

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
        file_state, anki_state, mock_converter, mock_key_map_instance
    )

    # 5. Verify Recovery
    # Should have called findNotes
    mock_invoke.assert_any_call("findNotes", query=f'"AnkiOps Key:{note_key}"')

    # Should have updated ID Map
    mock_key_map_instance.set_note.assert_called_with(note_key, recovered_note_id)


def test_create_injects_id(
    mock_key_map_instance,
    mock_invoke,
    anki_state,
    mock_converter,
    mock_key_map_class,
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

    mock_key_map_instance.get_note_id.return_value = None
    mock_key_map_instance.get_deck_id.return_value = 1
    mock_key_map_instance.get_deck_key.return_value = "deck-key-1"

    # Run Sync
    result, _ = _sync_file(
        file_state, anki_state, mock_converter, mock_key_map_instance
    )

    # Check changes
    change = result.changes[0]
    assert change.change_type == ChangeType.CREATE
    # Generated key from KeyMap.generate_key()
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
    km = KeyMap.load(tmp_path)
    km.set_note("n1", 100)
    km.close()

    db_path = tmp_path / KEY_MAP_DB
    assert db_path.exists()

    # Corrupt the DB file
    with open(db_path, "w") as f:
        f.write("corrupt data")

    # Attempt to load
    # It should detect corruption (SQLiteError), backup .db -> .db.corrupt,
    # and start fresh
    km = KeyMap.load(tmp_path)

    # Should be empty (fresh DB)
    assert km.get_note_id("n1") is None
    # Backup should exist
    assert (tmp_path / (KEY_MAP_DB + ".corrupt")).exists()
    km.close()

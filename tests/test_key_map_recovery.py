from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ankiops.config import KEY_MAP_DB
from ankiops.key_map import KeyMap
from ankiops.markdown_to_anki import _build_anki_actions, _sync_file
from ankiops.models import AnkiState, ChangeType, FileState, Note


@pytest.fixture
def mock_key_map_instance():
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
    """Test that a missing ID map entry is recovered via AnkiOps ID lookup."""

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

    # 3. Setup Anki to find the note via "AnkiOps ID"
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
                        "AnkiOps ID": {"value": note_key},
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
    mock_invoke.assert_any_call("findNotes", query=f'"AnkiOps ID:{note_key}"')

    # Should have updated ID Map
    mock_key_map_instance.set_note.assert_called_with(note_key, recovered_note_id)


def test_create_injects_id(
    mock_key_map_instance,
    mock_invoke,
    anki_state,
    mock_converter,
    mock_key_map_class,
):
    """Test that creating a note injects the AnkiOps ID field."""
    # Ensure KeyMap.generate_key returns a known value
    # mock_key_map_class is the class, we mocked generate_key on it

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
    assert fields["AnkiOps ID"] == "generated-key"


def test_corruption_recovery(tmp_path):
    """Test that if the DB is corrupt, it attempts to recover from JSON."""
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

    # Should be empty
    assert km.get_note_id("n1") is None
    assert (tmp_path / (KEY_MAP_DB + ".corrupt")).exists()
    km.close()

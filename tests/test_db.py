"""Tests for the Key Mapping system and recovery logic.

Includes:
1.  Basic CRUD operations (SQLite backed via SQLiteDbAdapter).
2.  Recovery from DB corruption.
3.  Key generation.
"""

import pytest

from ankiops.config import ANKIOPS_DB
from ankiops.db import SQLiteDbAdapter

# -- Basic CRUD Tests --------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Create a temporary SQLiteDbAdapter for testing."""
    return SQLiteDbAdapter.load(tmp_path)


def test_load_creates_tables(db, tmp_path):
    """Test that loading creates the necessary tables."""
    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()


def test_load_empty(db):
    """Loading from a non-existent file should return empty map."""
    assert db.get_note_id("any-key") is None
    assert db.get_deck_id("any-name") is None
    assert db.get_config("any-config") is None
    db.close()


def test_persistence(tmp_path):
    """Test that mappings and config persist across loads."""
    # First instance
    db1 = SQLiteDbAdapter.load(tmp_path)
    db1.set_deck("TestDeck", 100)
    db1.set_note("note-1", 200)
    db1.set_config("profile", "default")
    db1.save()
    db1.close()

    # Reload
    db2 = SQLiteDbAdapter.load(tmp_path)
    assert db2.get_deck_id("TestDeck") == 100
    assert db2.get_deck_name(100) == "TestDeck"
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
    """Test setting and getting deck name ↔ id mappings."""
    db.set_deck("MyDeck::Sub", 12345)
    assert db.get_deck_id("MyDeck::Sub") == 12345
    assert db.get_deck_name(12345) == "MyDeck::Sub"

    # Rename: same ID, new name should replace old entry
    db.set_deck("RenamedDeck", 12345)
    assert db.get_deck_id("RenamedDeck") == 12345
    assert db.get_deck_name(12345) == "RenamedDeck"
    assert db.get_deck_id("MyDeck::Sub") is None
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


def test_generate_key(db):
    """Test key generation."""
    key = db.generate_key()
    assert isinstance(key, str)
    assert len(key) > 10  # Reasonable length check for hex string


def test_corruption_recovery(tmp_path):
    """Test that if the DB is corrupt, it attempts to recover."""
    # Create a valid DB with one entry
    db = SQLiteDbAdapter.load(tmp_path)
    db.set_note("n1", 100)
    db.save()
    db.close()

    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()

    # Corrupt the DB file
    with open(db_path, "w") as f:
        f.write("corrupt data")

    # Attempt to load — should detect corruption, backup, and start fresh
    db = SQLiteDbAdapter.load(tmp_path)

    # Should be empty (fresh DB)
    assert db.get_note_id("n1") is None
    # Backup should exist
    assert (tmp_path / (ANKIOPS_DB + ".corrupt")).exists()
    db.close()

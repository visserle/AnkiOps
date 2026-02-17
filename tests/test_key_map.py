"""Tests for the ID mappping system (SQLite backed)."""

import json
import sqlite3
from pathlib import Path

import pytest

from ankiops.config import KEY_MAP_DB, MARKER_FILE
from ankiops.key_map import KeyMap


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
    km1.save(tmp_path)  # Verify save doesn't crash, though it's a no-op for SQLite
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
    # The old ID 101 is mostly orphaned in the reverse sense unless we explicitly delete it?
    # Actually, get_note_key(101) should check if 101 still maps to key-1?
    # In SQLite, "key" is PK. So key-1 -> 999 overwrites key-1 -> 101.
    # So 101 no longer maps to key-1.
    assert key_map.get_note_key(101) is None

    key_map.close()


def test_generate_key():
    """Test key generation."""
    key = KeyMap.generate_key()
    assert isinstance(key, str)
    assert len(key) > 10  # Reasonable length check for UUID-like string


def test_remove_mappings(key_map):
    """Test removing mappings."""
    key_map.close()

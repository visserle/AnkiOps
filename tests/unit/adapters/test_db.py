"""SQLite DB adapter contract tests (load, mapping, recovery)."""

from __future__ import annotations

import pytest

from ankiops.config import ANKIOPS_DB
from ankiops.db import SQLiteDbAdapter


@pytest.fixture
def db(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        yield adapter
    finally:
        adapter.close()


def test_load_creates_tables(db, tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()


def test_load_empty(db):
    assert db.get_note_id("any-key") is None
    assert db.get_deck_id("any-name") is None
    assert db.get_config("any-config") is None


def test_persistence(tmp_path):
    db1 = SQLiteDbAdapter.load(tmp_path)
    try:
        db1.set_deck("TestDeck", 100)
        db1.set_note("note-1", 200)
        db1.set_config("profile", "default")
        db1.save()
    finally:
        db1.close()

    db2 = SQLiteDbAdapter.load(tmp_path)
    try:
        assert db2.get_deck_id("TestDeck") == 100
        assert db2.get_deck_name(100) == "TestDeck"
        assert db2.get_note_id("note-1") == 200
        assert db2.get_config("profile") == "default"
    finally:
        db2.close()


def test_config_mapping(db):
    db.set_config("test-key", "test-value")
    assert db.get_config("test-key") == "test-value"

    db.set_config("test-key", "new-value")
    assert db.get_config("test-key") == "new-value"


def test_note_mapping(db):
    key = "test-note-key"
    note_id = 67890

    db.set_note(key, note_id)
    assert db.get_note_id(key) == note_id
    assert db.get_note_key(note_id) == key
    assert db.get_note_id("key-2") is None


def test_deck_mapping(db):
    db.set_deck("MyDeck::Sub", 12345)
    assert db.get_deck_id("MyDeck::Sub") == 12345
    assert db.get_deck_name(12345) == "MyDeck::Sub"

    db.set_deck("RenamedDeck", 12345)
    assert db.get_deck_id("RenamedDeck") == 12345
    assert db.get_deck_name(12345) == "RenamedDeck"
    assert db.get_deck_id("MyDeck::Sub") is None


def test_overwrite_mapping(db):
    db.set_note("key-1", 101)
    db.set_note("key-1", 999)

    assert db.get_note_id("key-1") == 999
    assert db.get_note_key(101) is None


def test_generate_key(db):
    key = db.generate_key()
    assert isinstance(key, str)
    assert len(key) > 10


def test_corruption_recovery(tmp_path):
    db = SQLiteDbAdapter.load(tmp_path)
    try:
        db.set_note("n1", 100)
        db.save()
    finally:
        db.close()

    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()
    db_path.write_text("corrupt data", encoding="utf-8")

    recovered = SQLiteDbAdapter.load(tmp_path)
    try:
        assert recovered.get_note_id("n1") is None
        assert (tmp_path / f"{ANKIOPS_DB}.corrupt").exists()
    finally:
        recovered.close()


def test_remove_note_by_id(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_note("k1", 101)
        assert adapter.get_note_id("k1") == 101

        adapter.remove_note_by_id(101)

        assert adapter.get_note_id("k1") is None
        assert adapter.get_note_key(101) is None
    finally:
        adapter.close()


def test_remove_deck(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_deck("Deck::Sub", 42)
        assert adapter.get_deck_id("Deck::Sub") == 42

        adapter.remove_deck("Deck::Sub")

        assert adapter.get_deck_id("Deck::Sub") is None
        assert adapter.get_deck_name(42) is None
    finally:
        adapter.close()


def test_same_id_reassignment_replaces_old_key(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_note("old-key", 500)
        adapter.set_note("new-key", 500)

        assert adapter.get_note_id("old-key") is None
        assert adapter.get_note_id("new-key") == 500
        assert adapter.get_note_key(500) == "new-key"
    finally:
        adapter.close()

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
    note_key = "test-note-key"
    note_id = 67890

    db.set_note(note_key, note_id)
    assert db.get_note_id(note_key) == note_id
    assert db.get_note_key(note_id) == note_key
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


def test_generate_note_key(db):
    note_key = db.generate_note_key()
    assert isinstance(note_key, str)
    assert len(note_key) > 10


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


def test_bulk_note_mapping_roundtrip(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_notes_bulk([("k1", 101), ("k2", 102), ("k3", 103)])

        assert adapter.get_note_ids_bulk(["k1", "k3", "missing"]) == {
            "k1": 101,
            "k3": 103,
        }
        assert adapter.get_note_keys_bulk([101, 103, 999]) == {
            101: "k1",
            103: "k3",
        }
    finally:
        adapter.close()


def test_bulk_note_mapping_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_notes_bulk(
            [
                ("k1", 1),
                ("k2", 2),
                ("k1", 3),  # note_key remapped to new note_id
                ("k3", 2),  # note_id remapped to new note_key
            ]
        )

        assert adapter.get_note_id("k1") == 3
        assert adapter.get_note_id("k2") is None
        assert adapter.get_note_id("k3") == 2
        assert adapter.get_note_key(1) is None
        assert adapter.get_note_key(2) == "k3"
        assert adapter.get_note_key(3) == "k1"
    finally:
        adapter.close()


def test_transaction_rolls_back_on_error(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with adapter.transaction():
                adapter.set_note("k1", 101)
                raise RuntimeError("boom")

        assert adapter.get_note_id("k1") is None
    finally:
        adapter.close()


def test_bulk_note_fingerprints_roundtrip(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_note_fingerprints_bulk(
            [
                ("k1", "md1", "a1"),
                ("k2", "md2", "a2"),
            ]
        )
        assert adapter.get_note_fingerprints_bulk(["k1", "k2", "missing"]) == {
            "k1": ("md1", "a1"),
            "k2": ("md2", "a2"),
        }
    finally:
        adapter.close()


def test_bulk_note_fingerprints_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_note_fingerprints_bulk(
            [
                ("k1", "old-md", "old-a"),
                ("k1", "new-md", "new-a"),
            ]
        )
        assert adapter.get_note_fingerprints_bulk(["k1"]) == {
            "k1": ("new-md", "new-a")
        }
    finally:
        adapter.close()


def test_remove_note_by_note_key_removes_fingerprint(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_note("k1", 101)
        adapter.set_note_fingerprints_bulk([("k1", "md", "a")])
        adapter.remove_note_by_note_key("k1")

        assert adapter.get_note_id("k1") is None
        assert adapter.get_note_fingerprints_bulk(["k1"]) == {}
    finally:
        adapter.close()


def test_markdown_media_cache_roundtrip_and_replace(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_markdown_media_cache_bulk(
            [("Deck.md", 10, 200, {"a.png", "b.png"})]
        )
        assert adapter.get_markdown_media_cache_bulk(["Deck.md"]) == {
            "Deck.md": (10, 200, {"a.png", "b.png"})
        }

        adapter.set_markdown_media_cache_bulk([("Deck.md", 11, 210, {"c.png"})])
        assert adapter.get_markdown_media_cache_bulk(["Deck.md"]) == {
            "Deck.md": (11, 210, {"c.png"})
        }

        adapter.remove_markdown_media_cache_by_paths(["Deck.md"])
        assert adapter.get_markdown_media_cache_bulk(["Deck.md"]) == {}
    finally:
        adapter.close()


def test_media_fingerprints_roundtrip_and_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_media_fingerprints_bulk(
            [
                ("a.png", 100, 1000, "d1", "a_d1.png"),
                ("a.png", 200, 1001, "d2", "a_d2.png"),
                ("b.png", 300, 1002, "d3", "b_d3.png"),
            ]
        )

        assert adapter.get_media_fingerprints_bulk(["a.png", "b.png"]) == {
            "a.png": (200, 1001, "d2", "a_d2.png"),
            "b.png": (300, 1002, "d3", "b_d3.png"),
        }

        adapter.remove_media_fingerprints_by_names(["a.png"])
        assert adapter.get_media_fingerprints_bulk(["a.png", "b.png"]) == {
            "b.png": (300, 1002, "d3", "b_d3.png")
        }
    finally:
        adapter.close()


def test_media_push_state_roundtrip_and_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.load(tmp_path)
    try:
        adapter.set_media_push_state_bulk(
            [("a.png", "d1"), ("a.png", "d2"), ("b.png", "d3")]
        )

        assert adapter.get_media_push_state_bulk(["a.png", "b.png"]) == {
            "a.png": "d2",
            "b.png": "d3",
        }

        adapter.remove_media_push_state_by_names(["a.png"])
        assert adapter.get_media_push_state_bulk(["a.png", "b.png"]) == {
            "b.png": "d3"
        }
    finally:
        adapter.close()

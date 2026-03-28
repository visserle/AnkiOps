"""SQLite DB adapter contract tests (open, mapping, recovery)."""

from __future__ import annotations

import sqlite3

import pytest

from ankiops.config import ANKIOPS_DB
from ankiops.db import SQLiteDbAdapter


@pytest.fixture
def db(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        yield adapter
    finally:
        adapter.close()


def test_load_creates_tables(db, tmp_path):
    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()
    assert db._conn.execute(
        "SELECT id, profile_name, note_type_sync_hash, note_type_names_signature "
        "FROM app_state"
    ).fetchall() == [(1, None, None, None)]


def test_load_empty(db):
    assert db.resolve_note_ids(["any-key"]).get("any-key") is None
    assert db.resolve_deck_id("any-name") is None
    assert db.get_profile_name() is None
    assert db.get_note_type_sync_state() is None


def test_persistence(tmp_path):
    db1 = SQLiteDbAdapter.open(tmp_path)
    try:
        db1.upsert_deck("TestDeck", 100)
        db1.upsert_note_links([("note-1", 200)])
        db1.set_profile_name("default")
        db1.set_note_type_sync_state("hash-1", "AnkiOpsQA")
    finally:
        db1.close()

    db2 = SQLiteDbAdapter.open(tmp_path)
    try:
        assert db2.resolve_deck_id("TestDeck") == 100
        assert db2.resolve_deck_name(100) == "TestDeck"
        assert db2.resolve_note_ids(["note-1"]).get("note-1") == 200
        assert db2.get_profile_name() == "default"
        assert db2.get_note_type_sync_state() == ("hash-1", "AnkiOpsQA")
    finally:
        db2.close()


def test_profile_mapping(db):
    db.set_profile_name("Default")
    assert db.get_profile_name() == "Default"

    db.set_profile_name("Work")
    assert db.get_profile_name() == "Work"


def test_note_type_sync_state_mapping(db):
    db.set_note_type_sync_state("hash-a", "AnkiOpsQA")
    assert db.get_note_type_sync_state() == ("hash-a", "AnkiOpsQA")

    db.set_note_type_sync_state("hash-b", "AnkiOpsQA,AnkiOpsCloze")
    assert db.get_note_type_sync_state() == ("hash-b", "AnkiOpsQA,AnkiOpsCloze")


def test_note_type_sync_state_requires_complete_pair(db):
    with pytest.raises(sqlite3.IntegrityError):
        db._write(
            "UPDATE app_state "
            "SET note_type_sync_hash = ?, note_type_names_signature = NULL "
            "WHERE id = 1",
            ("broken",),
        )


def test_note_mapping(db):
    note_key = "test-note-key"
    note_id = 67890

    db.upsert_note_links([(note_key, note_id)])
    assert db.resolve_note_ids([note_key]).get(note_key) == note_id
    assert db.resolve_note_keys([note_id]).get(note_id) == note_key
    assert db.resolve_note_ids(["key-2"]).get("key-2") is None


def test_deck_mapping(db):
    db.upsert_deck("MyDeck::Sub", 12345)
    assert db.resolve_deck_id("MyDeck::Sub") == 12345
    assert db.resolve_deck_name(12345) == "MyDeck::Sub"

    db.upsert_deck("RenamedDeck", 12345)
    assert db.resolve_deck_id("RenamedDeck") == 12345
    assert db.resolve_deck_name(12345) == "RenamedDeck"
    assert db.resolve_deck_id("MyDeck::Sub") is None


def test_overwrite_mapping(db):
    db.upsert_note_links([("key-1", 101)])
    db.upsert_note_links([("key-1", 999)])

    assert db.resolve_note_ids(["key-1"]).get("key-1") == 999
    assert db.resolve_note_keys([101]).get(101) is None


def test_generate_note_key(db):
    note_key = db.generate_note_key()
    assert isinstance(note_key, str)
    assert len(note_key) > 10


def test_corruption_recovery(tmp_path):
    db = SQLiteDbAdapter.open(tmp_path)
    try:
        db.upsert_note_links([("n1", 100)])
    finally:
        db.close()

    db_path = tmp_path / ANKIOPS_DB
    assert db_path.exists()
    db_path.write_text("corrupt data", encoding="utf-8")

    recovered = SQLiteDbAdapter.open(tmp_path)
    try:
        assert recovered.resolve_note_ids(["n1"]).get("n1") is None
        assert (tmp_path / f"{ANKIOPS_DB}.corrupt").exists()
    finally:
        recovered.close()


def test_schema_mismatch_recovery(tmp_path):
    db_path = tmp_path / ANKIOPS_DB

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE legacy_state (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    recovered = SQLiteDbAdapter.open(tmp_path)
    try:
        assert recovered.get_profile_name() is None
        assert recovered.get_note_type_sync_state() is None
    finally:
        recovered.close()

    assert (tmp_path / f"{ANKIOPS_DB}.corrupt").exists()


def test_remove_note_by_id(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        assert adapter.resolve_note_ids(["k1"]).get("k1") == 101

        adapter.delete_note_link_by_id(101)

        assert adapter.resolve_note_ids(["k1"]).get("k1") is None
        assert adapter.resolve_note_keys([101]).get(101) is None
    finally:
        adapter.close()


def test_delete_deck(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_deck("Deck::Sub", 42)
        assert adapter.resolve_deck_id("Deck::Sub") == 42

        adapter.delete_deck("Deck::Sub")

        assert adapter.resolve_deck_id("Deck::Sub") is None
        assert adapter.resolve_deck_name(42) is None
    finally:
        adapter.close()


def test_same_id_reassignment_replaces_old_key(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("old-key", 500)])
        adapter.upsert_note_links([("new-key", 500)])

        assert adapter.resolve_note_ids(["old-key"]).get("old-key") is None
        assert adapter.resolve_note_ids(["new-key"]).get("new-key") == 500
        assert adapter.resolve_note_keys([500]).get(500) == "new-key"
    finally:
        adapter.close()


def test_bulk_note_mapping_roundtrip(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101), ("k2", 102), ("k3", 103)])

        assert adapter.resolve_note_ids(["k1", "k3", "missing"]) == {
            "k1": 101,
            "k3": 103,
        }
        assert adapter.resolve_note_keys([101, 103, 999]) == {
            101: "k1",
            103: "k3",
        }
    finally:
        adapter.close()


def test_bulk_note_mapping_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links(
            [
                ("k1", 1),
                ("k2", 2),
                ("k1", 3),  # note_key remapped to new note_id
                ("k3", 2),  # note_id remapped to new note_key
            ]
        )

        assert adapter.resolve_note_ids(["k1"]).get("k1") == 3
        assert adapter.resolve_note_ids(["k2"]).get("k2") is None
        assert adapter.resolve_note_ids(["k3"]).get("k3") == 2
        assert adapter.resolve_note_keys([1]).get(1) is None
        assert adapter.resolve_note_keys([2]).get(2) == "k3"
        assert adapter.resolve_note_keys([3]).get(3) == "k1"
    finally:
        adapter.close()


def test_transaction_rolls_back_on_error(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with adapter.write_tx():
                adapter.upsert_note_links([("k1", 101)])
                raise RuntimeError("boom")

        assert adapter.resolve_note_ids(["k1"]).get("k1") is None
    finally:
        adapter.close()


def test_bulk_import_note_fingerprints_roundtrip(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101), ("k2", 102)])
        adapter.upsert_import_hashes(
            [
                ("k1", "md1", "a1"),
                ("k2", "md2", "a2"),
            ]
        )
        assert adapter.resolve_import_hashes(["k1", "k2", "missing"]) == {
            "k1": ("md1", "a1"),
            "k2": ("md2", "a2"),
        }
    finally:
        adapter.close()


def test_bulk_export_note_fingerprints_roundtrip(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101), ("k2", 102)])
        adapter.upsert_export_hashes(
            [
                ("k1", "md1", "a1"),
                ("k2", "md2", "a2"),
            ]
        )
        assert adapter.resolve_export_hashes(["k1", "k2", "missing"]) == {
            "k1": ("md1", "a1"),
            "k2": ("md2", "a2"),
        }
    finally:
        adapter.close()


def test_import_note_fingerprints_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        adapter.upsert_import_hashes(
            [
                ("k1", "old-md", "old-a"),
                ("k1", "new-md", "new-a"),
            ]
        )
        assert adapter.resolve_import_hashes(["k1"]) == {"k1": ("new-md", "new-a")}
    finally:
        adapter.close()


def test_export_note_fingerprints_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        adapter.upsert_export_hashes(
            [
                ("k1", "old-md", "old-a"),
                ("k1", "new-md", "new-a"),
            ]
        )
        assert adapter.resolve_export_hashes(["k1"]) == {"k1": ("new-md", "new-a")}
    finally:
        adapter.close()


def test_remove_note_by_note_key_removes_directional_fingerprints(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        adapter.upsert_import_hashes([("k1", "imd", "ia")])
        adapter.upsert_export_hashes([("k1", "emd", "ea")])
        adapter.delete_note_links_by_keys(["k1"])

        assert adapter.resolve_note_ids(["k1"]).get("k1") is None
        assert adapter.resolve_import_hashes(["k1"]) == {}
        assert adapter.resolve_export_hashes(["k1"]) == {}
    finally:
        adapter.close()


def test_unknown_note_key_import_fingerprint_is_rejected(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            adapter.upsert_import_hashes([("stale", "md2", "a2")])
    finally:
        adapter.close()


def test_unknown_note_key_export_fingerprint_is_rejected(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            adapter.upsert_export_hashes([("stale", "md2", "a2")])
    finally:
        adapter.close()


def test_clear_import_hashes_does_not_clear_export_hashes(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        adapter.upsert_import_hashes([("k1", "imd1", "ia1")])
        adapter.upsert_export_hashes([("k1", "emd1", "ea1")])

        adapter.clear_import_hashes(["k1"])

        assert adapter.resolve_import_hashes(["k1"]) == {}
        assert adapter.resolve_export_hashes(["k1"]) == {"k1": ("emd1", "ea1")}
    finally:
        adapter.close()


def test_clear_export_hashes_does_not_clear_import_hashes(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_note_links([("k1", 101)])
        adapter.upsert_import_hashes([("k1", "imd1", "ia1")])
        adapter.upsert_export_hashes([("k1", "emd1", "ea1")])

        adapter.clear_export_hashes(["k1"])

        assert adapter.resolve_import_hashes(["k1"]) == {"k1": ("imd1", "ia1")}
        assert adapter.resolve_export_hashes(["k1"]) == {}
    finally:
        adapter.close()


def test_markdown_media_cache_roundtrip_and_replace(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_markdown_media_cache(
            [("Deck.md", 10, 200, {"a.png", "b.png"})]
        )
        assert adapter.resolve_markdown_media_cache(["Deck.md"]) == {
            "Deck.md": (10, 200, {"a.png", "b.png"})
        }

        adapter.upsert_markdown_media_cache([("Deck.md", 11, 210, {"c.png"})])
        assert adapter.resolve_markdown_media_cache(["Deck.md"]) == {
            "Deck.md": (11, 210, {"c.png"})
        }

        adapter.delete_markdown_media_cache(["Deck.md"])
        assert adapter.resolve_markdown_media_cache(["Deck.md"]) == {}
    finally:
        adapter.close()


def test_prune_markdown_media_cache_removes_stale_paths(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_markdown_media_cache(
            [
                ("DeckA.md", 10, 200, {"a.png"}),
                ("DeckB.md", 20, 300, {"b.png"}),
            ]
        )

        removed = adapter.prune_markdown_media_cache({"DeckB.md"})

        assert removed == 1
        assert adapter.resolve_markdown_media_cache(["DeckA.md", "DeckB.md"]) == {
            "DeckB.md": (20, 300, {"b.png"})
        }
    finally:
        adapter.close()


def test_media_fingerprints_roundtrip_and_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_media_fingerprints(
            [
                ("a.png", 100, 1000, "d1", "a_d1.png"),
                ("a.png", 200, 1001, "d2", "a_d2.png"),
                ("b.png", 300, 1002, "d3", "b_d3.png"),
            ]
        )

        assert adapter.resolve_media_fingerprints(["a.png", "b.png"]) == {
            "a.png": (200, 1001, "d2", "a_d2.png"),
            "b.png": (300, 1002, "d3", "b_d3.png"),
        }

        adapter.delete_media_state(["a.png"])
        assert adapter.resolve_media_fingerprints(["a.png", "b.png"]) == {
            "b.png": (300, 1002, "d3", "b_d3.png")
        }
    finally:
        adapter.close()


def test_media_push_state_roundtrip_and_last_write_wins(tmp_path):
    adapter = SQLiteDbAdapter.open(tmp_path)
    try:
        adapter.upsert_media_fingerprints(
            [
                ("a.png", 100, 1000, "d1", "a_d1.png"),
                ("b.png", 300, 1002, "d3", "b_d3.png"),
            ]
        )
        adapter.upsert_media_push_digests(
            [("a.png", "d1"), ("a.png", "d2"), ("b.png", "d3")]
        )

        assert adapter.resolve_media_push_digests(["a.png", "b.png"]) == {
            "a.png": "d2",
            "b.png": "d3",
        }

        adapter.clear_media_push_digests(["a.png"])
        assert adapter.resolve_media_push_digests(["a.png", "b.png"]) == {"b.png": "d3"}
    finally:
        adapter.close()

"""Tests for collection serialization/deserialization."""

import pytest

from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    serialize_collection_to_json,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter


@pytest.fixture
def collection(tmp_path, fs, monkeypatch):
    """Create a minimal collection directory with one deck file."""
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: tmp_path)

    # Initialize DB
    db = SQLiteDbAdapter.load(tmp_path)
    db.set_config("profile", "test")
    db.save()
    db.close()

    # Eject note types
    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)

    # Create a deck file
    deck_md = tmp_path / "TestDeck.md"
    deck_md.write_text(
        ""
        "<!-- note_key: nk-1 -->\n"
        "Q: What is 2+2?\n"
        "A: 4\n\n"
        "---\n\n"
        "<!-- note_key: nk-2 -->\n"
        "Q: What is 3+3?\n"
        "A: 6"
    )

    return tmp_path


class TestSerialize:
    """Test serializing a collection to JSON."""

    def test_basic_serialize(self, collection, tmp_path, monkeypatch):
        monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection)
        output = tmp_path / "out.json"
        result = serialize_collection_to_json(collection, output)

        assert len(result["decks"]) == 1
        assert len(result["decks"][0]["notes"]) == 2
        assert result["decks"][0]["notes"][0]["note_key"] == "nk-1"


class TestDeserialize:
    """Test deserializing from JSON to markdown files."""

    def test_roundtrip(self, collection, tmp_path, monkeypatch):
        """Serialize then deserialize should produce equivalent content."""
        monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection)
        monkeypatch.setattr(
            "ankiops.config.get_note_types_dir", lambda: collection / "note_types"
        )
        monkeypatch.setattr(
            "ankiops.collection_serializer.get_collection_dir", lambda: collection
        )
        monkeypatch.setattr(
            "ankiops.collection_serializer.get_note_types_dir",
            lambda: collection / "note_types",
        )
        json_file = tmp_path / "export.json"
        serialize_collection_to_json(collection, json_file)

        # Deserialize to a fresh directory
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: fresh_dir)
        monkeypatch.setattr(
            "ankiops.collection_serializer.get_collection_dir", lambda: fresh_dir
        )

        # Copy note types
        fs = FileSystemAdapter()
        note_types_dst = fresh_dir / "note_types"
        fs.eject_builtin_note_types(note_types_dst)
        monkeypatch.setattr("ankiops.config.get_note_types_dir", lambda: note_types_dst)
        monkeypatch.setattr(
            "ankiops.collection_serializer.get_note_types_dir", lambda: note_types_dst
        )

        deserialize_collection_from_json(json_file, overwrite=True)

        # Should have created a deck file
        md_files = list(fresh_dir.glob("*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "What is 2+2?" in content
        assert "What is 3+3?" in content

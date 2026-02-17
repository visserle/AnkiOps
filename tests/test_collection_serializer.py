"""Tests for collection serialization and deserialization."""

import json
import os
from pathlib import Path

from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    serialize_collection_to_json,
)
from ankiops.config import MARKER_FILE
from ankiops.models import FileState, Note


class TestDeserialization:
    """Test collection deserialization."""

    def create_test_json(self, tmp_path):
        """Helper to create a test JSON file."""
        serialized_file = tmp_path / "test.json"

        serialized_data = {
            "collection": {
                "serialized_at": "2024-01-01T00:00:00Z",
            },
            "decks": [
                {
                    "deck_key": "deck-key-1",
                    "name": "Test Deck",
                    "notes": [
                        {
                            "note_key": "key-111",
                            "fields": {
                                "Question": "What is this?",
                                "Answer": "An image",
                            },
                        }
                    ],
                }
            ],
        }

        serialized_file.write_text(json.dumps(serialized_data, indent=2))
        return serialized_file

    def test_deserialize_with_ignore_ids(self, tmp_path):
        """Test deserialization with no_ids=True skips writing key comments."""
        serialized_file = self.create_test_json(tmp_path)
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Create marker
        (collection_dir / MARKER_FILE).touch()

        # Deserialize with no_ids=True, ensuring we are in the target dir
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(
                serialized_file, overwrite=True, no_ids=True
            )
        finally:
            os.chdir(original_cwd)

        # Verify markdown file was created
        deck_file = collection_dir / "Test Deck.md"
        assert deck_file.exists()

        # Verify NO ID comments appear in the markdown
        content = deck_file.read_text()
        assert "<!-- deck_key:" not in content
        assert "<!-- note_key:" not in content
        assert "<!-- uuid:" not in content

        # Verify note content is still present
        assert "What is this?" in content
        assert "An image" in content

    def test_deserialize_with_keys(self, tmp_path):
        """Test deserialization writes keys by default."""
        serialized_file = self.create_test_json(tmp_path)
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()
        (collection_dir / MARKER_FILE).touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(
                serialized_file, overwrite=True, no_ids=False
            )
        finally:
            os.chdir(original_cwd)

        deck_file = collection_dir / "Test Deck.md"
        assert deck_file.exists()
        content = deck_file.read_text()

        # Verify keys are present
        assert "<!-- deck_key: deck-key-1 -->" in content
        assert "<!-- note_key: key-111 -->" in content


class TestSerialization:
    """Test collection serialization."""

    def test_serialize_uses_keys(self, tmp_path):
        """Verify serialization uses deck_key and note_key."""
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()
        (collection_dir / MARKER_FILE).touch()

        # Create a markdown file with keys
        deck_file = collection_dir / "Source Deck.md"
        deck_file.write_text(
            "<!-- deck_key: deck-key-ABC -->\n"
            "<!-- note_key: note-key-123 -->\n"
            "Q: Question\nA: Answer"
        )

        output_file = tmp_path / "output.json"

        serialize_collection_to_json(collection_dir, output_file)

        assert output_file.exists()
        data = json.loads(output_file.read_text())

        assert len(data["decks"]) == 1
        deck = data["decks"][0]
        assert deck["deck_key"] == "deck-key-ABC"
        assert len(deck["notes"]) == 1
        note = deck["notes"][0]
        assert note["note_key"] == "note-key-123"
        assert note["fields"]["Question"] == "Question"

    def test_deserialize_ignores_uuid(self, tmp_path):
        """Test that legacy 'uuid' field is ignored if 'note_key' is missing."""
        serialized_file = tmp_path / "legacy.json"

        # JSON with 'uuid' but no 'note_key'
        data = {
            "collection": {"serialized_at": "2024-01-01"},
            "decks": [
                {
                    "name": "Legacy Deck",
                    "uuid": "legacy-deck-uuid",
                    "notes": [
                        {
                            "uuid": "legacy-note-uuid",
                            "fields": {"Front": "F", "Back": "B"},
                        }
                    ],
                }
            ],
        }
        serialized_file.write_text(json.dumps(data))

        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()
        (collection_dir / MARKER_FILE).touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file, overwrite=True)
        finally:
            os.chdir(original_cwd)

        deck_file = collection_dir / "Legacy Deck.md"
        assert deck_file.exists()
        content = deck_file.read_text()

        # UUIDs should NOT be in the output as key comments
        assert "<!-- deck_key: legacy-deck-uuid -->" not in content
        assert "<!-- note_key: legacy-note-uuid -->" not in content

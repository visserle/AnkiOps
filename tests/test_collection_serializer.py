"""Tests for collection serialization and deserialization."""

import json
import os

from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    extract_media_references,
    serialize_collection_to_json,
)





class TestMediaReferences:
    """Test media reference extraction and updating."""

    def test_extract_markdown_images(self):
        """Test extracting markdown image references."""
        text = "Some text ![alt](image.png) more text ![](another.jpg)"
        refs = extract_media_references(text)
        assert "image.png" in refs
        assert "another.jpg" in refs

    def test_extract_media_prefix(self):
        """Test extracting references with media/ prefix."""
        text = "![](media/image.png) and [sound:media/audio.mp3]"
        refs = extract_media_references(text)
        # Should strip media/ prefix
        assert "image.png" in refs
        assert "audio.mp3" in refs
        assert "media/image.png" not in refs

    def test_extract_sound_tags(self):
        """Test extracting Anki sound tags."""
        text = "Text with [sound:audio.mp3] and [sound:music.ogg]"
        refs = extract_media_references(text)
        assert "audio.mp3" in refs
        assert "music.ogg" in refs

    def test_extract_html_img(self):
        """Test extracting HTML img tags."""
        text = '<img src="image.png"> and <img src="photo.jpg">'
        refs = extract_media_references(text)
        assert "image.png" in refs
        assert "photo.jpg" in refs




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
                    "deck_id": "1234567890",
                    "name": "Test Deck",
                    "notes": [
                        {
                            "note_id": "1111111111",
                            "fields": {
                                "Question": "What is this? ![](media/test.png)",
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
        """Test deserialization with ignore_ids skips writing ID comments."""
        serialized_file = self.create_test_json(tmp_path)
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Deserialize with no_ids=True
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file, no_ids=True)
        finally:
            os.chdir(original_cwd)

        # Verify markdown file was created
        deck_file = collection_dir / "Test Deck.md"
        assert deck_file.exists()

        # Verify NO ID comments appear in the markdown
        content = deck_file.read_text()
        assert "<!-- deck_id:" not in content
        assert "<!-- note_id:" not in content

        # Verify note content is still present
        assert "What is this?" in content
        assert "An image" in content


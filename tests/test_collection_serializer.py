"""Tests for collection serialization and deserialization."""

import hashlib
import json
import os
import zipfile

from ankiops.collection_serializer import (
    compute_file_hash,
    compute_zipfile_hash,
    deserialize_collection_from_json,
    extract_media_references,
    serialize_collection_to_json,
    update_media_references,
)


class TestHashFunctions:
    """Test file hashing functions."""

    def test_compute_file_hash(self, tmp_path):
        """Test computing hash of a regular file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = compute_file_hash(test_file)

        # Verify it's a valid SHA256 hash (64 hex characters)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

        # Verify same content produces same hash
        test_file2 = tmp_path / "test2.txt"
        test_file2.write_text("Hello, World!")
        assert compute_file_hash(test_file2) == hash_result

        # Verify different content produces different hash
        test_file3 = tmp_path / "test3.txt"
        test_file3.write_text("Different content")
        assert compute_file_hash(test_file3) != hash_result

    def test_compute_zipfile_hash(self, tmp_path):
        """Test computing hash of a file in a ZIP archive."""
        zip_file = tmp_path / "test.zip"
        test_content = b"Hello, World!"

        with zipfile.ZipFile(zip_file, "w") as zipf:
            zipf.writestr("test.txt", test_content)

        with zipfile.ZipFile(zip_file, "r") as zipf:
            hash_result = compute_zipfile_hash(zipf, "test.txt")

        # Verify it matches the expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert hash_result == expected_hash


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

    def test_update_markdown_image_references(self):
        """Test updating markdown image references."""
        text = "Some text ![alt](media/image.png) more text"
        rename_map = {"image.png": "image_1.png"}

        updated = update_media_references(text, rename_map)

        assert "media/image_1.png" in updated
        assert "media/image.png" not in updated

    def test_update_sound_references(self):
        """Test updating Anki sound tag references."""
        text = "Text with [sound:audio.mp3] here"
        rename_map = {"audio.mp3": "audio_1.mp3"}

        updated = update_media_references(text, rename_map)

        assert "[sound:audio_1.mp3]" in updated
        assert "[sound:audio.mp3]" not in updated

    def test_update_html_img_references(self):
        """Test updating HTML img tag references."""
        text = '<img src="media/image.png" alt="test">'
        rename_map = {"image.png": "image_1.png"}

        updated = update_media_references(text, rename_map)

        assert "media/image_1.png" in updated
        assert "media/image.png" not in updated

    def test_update_no_changes_when_no_matches(self):
        """Test that text is unchanged when no references match."""
        text = "![](image.png) and [sound:audio.mp3]"
        rename_map = {"other.png": "other_1.png"}

        updated = update_media_references(text, rename_map)

        assert updated == text


class TestDeserializeMediaConflicts:
    """Test media file conflict handling during deserialization."""

    def create_test_serialized(self, tmp_path, media_content):
        """Helper to create a test serialized file with media."""
        serialized_file = tmp_path / "test.zip"

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

        with zipfile.ZipFile(serialized_file, "w") as zipf:
            # Add JSON
            zipf.writestr("collection.json", json.dumps(serialized_data, indent=2))
            # Add media file
            zipf.writestr("media/test.png", media_content)

        return serialized_file

    def test_deserialize_with_no_existing_media(self, tmp_path):
        """Test normal deserialization with no conflicts."""
        serialized_file = self.create_test_serialized(tmp_path, b"image data")
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Change to collection directory and deserialize
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file)
        finally:
            os.chdir(original_cwd)

        # Verify media file was extracted
        media_file = collection_dir / "media" / "AnkiOpsMedia" / "test.png"
        assert media_file.exists()
        assert media_file.read_bytes() == b"image data"

        # Verify note references unchanged
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test.png)" in content

    def test_deserialize_with_identical_existing_media(self, tmp_path):
        """Test deserialization skips identical media files."""
        serialized_file = self.create_test_serialized(tmp_path, b"image data")
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Create existing media file with same content
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        existing_file = media_dir / "test.png"
        existing_file.write_bytes(b"image data")

        # Deserialize
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file)
        finally:
            os.chdir(original_cwd)

        # Verify file still has original content (not overwritten)
        assert existing_file.read_bytes() == b"image data"

        # Verify note references unchanged
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test.png)" in content

    def test_deserialize_with_different_existing_media(self, tmp_path):
        """Test deserialization renames conflicting media files."""
        serialized_file = self.create_test_serialized(tmp_path, b"new image data")
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Create existing media file with different content
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        existing_file = media_dir / "test.png"
        existing_file.write_bytes(b"old image data")

        # Deserialize
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file)
        finally:
            os.chdir(original_cwd)

        # Verify original file unchanged
        assert existing_file.read_bytes() == b"old image data"

        # Verify new file was renamed
        renamed_file = media_dir / "test_1.png"
        assert renamed_file.exists()
        assert renamed_file.read_bytes() == b"new image data"

        # Verify note references updated to renamed file
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test_1.png)" in content
        assert "![](media/test.png)" not in content

    def test_deserialize_with_multiple_conflicts(self, tmp_path):
        """Test deserialization handles multiple renamed files."""
        serialized_file = self.create_test_serialized(tmp_path, b"newest data")
        collection_dir = tmp_path / "collection"
        collection_dir.mkdir()

        # Create existing media files
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "test.png").write_bytes(b"original data")
        (media_dir / "test_1.png").write_bytes(b"first rename")

        # Deserialize
        original_cwd = os.getcwd()
        try:
            os.chdir(collection_dir)
            deserialize_collection_from_json(serialized_file)
        finally:
            os.chdir(original_cwd)

        # Verify files unchanged
        assert (media_dir / "test.png").read_bytes() == b"original data"
        assert (media_dir / "test_1.png").read_bytes() == b"first rename"

        # Verify new file got next available number
        renamed_file = media_dir / "test_2.png"
        assert renamed_file.exists()
        assert renamed_file.read_bytes() == b"newest data"

        # Verify note references updated
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test_2.png)" in content

    def test_deserialize_with_ignore_ids(self, tmp_path):
        """Test deserialization with ignore_ids skips writing ID comments."""
        serialized_file = self.create_test_serialized(tmp_path, b"image data")
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


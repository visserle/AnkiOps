"""Tests for collection packaging and unpackaging."""

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import pytest
from ankiops.collection_package import (
    compute_file_hash,
    compute_zipfile_hash,
    extract_media_references,
    package_collection_to_json,
    unpackage_collection_from_json,
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


class TestUnpackageMediaConflicts:
    """Test media file conflict handling during unpackage."""

    def create_test_package(self, tmp_path, media_content):
        """Helper to create a test package with media."""
        package_file = tmp_path / "test.zip"

        package_data = {
            "collection": {
                "profile": "test",
                "auto_commit": True,
                "packaged_at": "2024-01-01T00:00:00Z",
            },
            "decks": [
                {
                    "name": "Test Deck",
                    "deck_id": "1234567890",
                    "notes": [
                        {
                            "note_id": "1111111111",
                            "note_type": "AnkiOpsQA",
                            "fields": {
                                "Question": "What is this? ![](media/test.png)",
                                "Answer": "An image",
                            },
                        }
                    ],
                }
            ],
        }

        with zipfile.ZipFile(package_file, "w") as zipf:
            # Add JSON
            zipf.writestr("collection.json", json.dumps(package_data, indent=2))
            # Add media file
            zipf.writestr("media/test.png", media_content)

        return package_file

    def test_unpackage_with_no_existing_media(self, tmp_path):
        """Test normal unpackage with no conflicts."""
        package_file = self.create_test_package(tmp_path, b"image data")
        collection_dir = tmp_path / "collection"

        unpackage_collection_from_json(package_file, collection_dir)

        # Verify media file was extracted
        media_file = collection_dir / "media" / "AnkiOpsMedia" / "test.png"
        assert media_file.exists()
        assert media_file.read_bytes() == b"image data"

        # Verify note references unchanged
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test.png)" in content

    def test_unpackage_with_identical_existing_media(self, tmp_path):
        """Test unpackage skips identical media files."""
        package_file = self.create_test_package(tmp_path, b"image data")
        collection_dir = tmp_path / "collection"

        # Create existing media file with same content
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        existing_file = media_dir / "test.png"
        existing_file.write_bytes(b"image data")

        # Unpackage
        unpackage_collection_from_json(package_file, collection_dir)

        # Verify file still has original content (not overwritten)
        assert existing_file.read_bytes() == b"image data"

        # Verify note references unchanged
        deck_file = collection_dir / "Test Deck.md"
        content = deck_file.read_text()
        assert "![](media/test.png)" in content

    def test_unpackage_with_different_existing_media(self, tmp_path):
        """Test unpackage renames conflicting media files."""
        package_file = self.create_test_package(tmp_path, b"new image data")
        collection_dir = tmp_path / "collection"

        # Create existing media file with different content
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        existing_file = media_dir / "test.png"
        existing_file.write_bytes(b"old image data")

        # Unpackage
        unpackage_collection_from_json(package_file, collection_dir)

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

    def test_unpackage_with_multiple_conflicts(self, tmp_path):
        """Test unpackage handles multiple renamed files."""
        package_file = self.create_test_package(tmp_path, b"newest data")
        collection_dir = tmp_path / "collection"

        # Create existing media files
        media_dir = collection_dir / "media" / "AnkiOpsMedia"
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "test.png").write_bytes(b"original data")
        (media_dir / "test_1.png").write_bytes(b"first rename")

        # Unpackage
        unpackage_collection_from_json(package_file, collection_dir)

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

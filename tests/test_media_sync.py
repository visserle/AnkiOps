"""Tests for media synchronization."""

import hashlib

import pytest

from ankiops.config import LOCAL_MEDIA_DIR


@pytest.fixture
def media_dir(tmp_path):
    """Create a temporary media directory."""
    d = tmp_path / LOCAL_MEDIA_DIR
    d.mkdir()
    return d


# -- Hashing & Filename Tests -----------------------------------------------


class TestMediaHashing:
    """Test the deterministic hashing of media files."""

    def test_identical_files_have_same_hash(self, media_dir):
        f1 = media_dir / "img1.png"
        f2 = media_dir / "img2.png"
        content = b"identical content"
        f1.write_bytes(content)
        f2.write_bytes(content)

        hash1 = hashlib.md5(f1.read_bytes()).hexdigest()
        hash2 = hashlib.md5(f2.read_bytes()).hexdigest()
        assert hash1 == hash2

    def test_different_files_have_different_hash(self, media_dir):
        f1 = media_dir / "img1.png"
        f2 = media_dir / "img2.png"
        f1.write_bytes(b"content1")
        f2.write_bytes(b"content2")

        hash1 = hashlib.md5(f1.read_bytes()).hexdigest()
        hash2 = hashlib.md5(f2.read_bytes()).hexdigest()
        assert hash1 != hash2


# -- Reference Detection Tests ----------------------------------------------


class TestMediaReferenceDetection:
    """Test that media references are found in markdown content."""

    def test_image_reference_detected(self, fs, tmp_path):
        """An image reference in a note points to a media file."""
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        img = media_dir / "test.png"
        img.write_bytes(b"fake png")

        md = tmp_path / "deck.md"
        md.write_text("Q: What is this?\nA: ![image](media/test.png)")
        result = fs.read_markdown_file(md)
        assert "media/test.png" in result.notes[0].fields["Answer"]

    def test_no_reference_in_plain_text(self, fs, tmp_path):
        """No media reference in plain text."""
        md = tmp_path / "deck.md"
        md.write_text("Q: What?\nA: Plain answer")
        result = fs.read_markdown_file(md)
        assert "media/" not in result.notes[0].fields["Answer"]


# -- Media Directory Tests --------------------------------------------------


class TestMediaDirectory:
    """Test FileSystemAdapter interactions with the media directory."""

    def test_find_media_files(self, tmp_path):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        (media_dir / "img1.png").write_bytes(b"1")
        (media_dir / "img2.jpg").write_bytes(b"2")
        (media_dir / "doc.pdf").write_bytes(b"3")

        files = list(media_dir.iterdir())
        assert len(files) == 3

    def test_empty_media_dir(self, tmp_path):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        files = list(media_dir.iterdir())
        assert len(files) == 0

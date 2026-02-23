"""Tests for media synchronization."""

import hashlib

import pytest

from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.sync_media import _extract_media_references


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

    def test_extract_media_references_parentheses(self):
        
        # Regular link
        assert _extract_media_references("![img](media/a.jpg)") == {"a.jpg"}
        # Link with parentheses inside it
        assert _extract_media_references("![img](media/a(b).jpg)") == {"a(b).jpg"}
        assert _extract_media_references("![img](<media/a(b).jpg>)") == {"a(b).jpg"}
        
    def test_extract_media_references_url_encoding(self):
        
        assert _extract_media_references("![img](media/a%20b.jpg)") == {"a b.jpg"}
        assert _extract_media_references("![img](media/a%27s.jpg)") == {"a's.jpg"}

    def test_extract_media_references_html(self):
        
        # HTML tags shouldn't get messed up if they have parentheses or spaces
        assert _extract_media_references('<img src="a(b).jpg">') == {"a(b).jpg"}
        assert _extract_media_references('<img src="a b.jpg">') == {"a b.jpg"}
        assert _extract_media_references('<img src="media/a%20b.jpg">') == {"a b.jpg"}

    def test_extract_media_references_ignores_external(self):
        
        assert _extract_media_references('<img src="http://example.com/a.jpg">') == set()
        assert _extract_media_references('![img](https://example.com/a.jpg)') == set()

    def test_update_media_references_parentheses_and_html(self, fs, tmp_path):
        md_dir = tmp_path / "collection"
        md_dir.mkdir()
        md_file = md_dir / "deck.md"
        
        original_content = (
            "1. ![img](media/a(b).jpg)\n"
            "2. <img src=\"a b.jpg\">\n"
            "3. [sound:a%20b.mp3]\n"
        )
        md_file.write_text(original_content)
        
        rename_map = {
            "media/a(b).jpg": "media/hashed1.jpg",
            "a b.jpg": "media/hashed2.jpg",
            "a b.mp3": "hashed3.mp3",
        }
        
        updated_count = fs.update_media_references(md_dir, rename_map)
        assert updated_count == 1
        
        new_content = md_file.read_text()
        assert "![img](<media/hashed1.jpg>)" in new_content
        assert "<img src=\"media/hashed2.jpg\">" in new_content
        assert "[sound:hashed3.mp3]" in new_content


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

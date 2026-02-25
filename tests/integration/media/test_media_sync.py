"""Media synchronization tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from blake3 import blake3

from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.db import SQLiteDbAdapter
from ankiops.sync_media import _extract_media_references, sync_media_to_anki


class _FakeMediaAnki:
    def __init__(self, media_dir: Path):
        self._media_dir = media_dir
        self.push_count = 0

    def get_media_dir(self) -> Path:
        return self._media_dir

    def push_media(self, local_path: Path, remote_filename: str) -> None:
        self.push_count += 1
        target = self._media_dir / remote_filename
        target.write_bytes(local_path.read_bytes())


def _blake3(path: Path) -> str:
    return blake3(path.read_bytes()).hexdigest(length=4)


@pytest.fixture
def media_dir(tmp_path):
    directory = tmp_path / LOCAL_MEDIA_DIR
    directory.mkdir()
    return directory


class TestMediaHashing:
    """Hashing should be content-based and deterministic."""

    def test_identical_files_have_same_hash(self, media_dir):
        first = media_dir / "img1.png"
        second = media_dir / "img2.png"
        content = b"identical content"
        first.write_bytes(content)
        second.write_bytes(content)

        assert _blake3(first) == _blake3(second)

    def test_different_files_have_different_hash(self, media_dir):
        first = media_dir / "img1.png"
        second = media_dir / "img2.png"
        first.write_bytes(b"content1")
        second.write_bytes(b"content2")

        assert _blake3(first) != _blake3(second)


class TestMediaReferenceDetection:
    """References in markdown and HTML should resolve to local media names only."""

    def test_image_reference_detected(self, fs, tmp_path):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        (media_dir / "test.png").write_bytes(b"fake png")

        md = tmp_path / "deck.md"
        md.write_text("Q: What is this?\nA: ![image](media/test.png)", encoding="utf-8")
        result = fs.read_markdown_file(md)
        assert "media/test.png" in result.notes[0].fields["Answer"]

    def test_no_reference_in_plain_text(self, fs, tmp_path):
        md = tmp_path / "deck.md"
        md.write_text("Q: What?\nA: Plain answer", encoding="utf-8")
        result = fs.read_markdown_file(md)
        assert "media/" not in result.notes[0].fields["Answer"]

    def test_extract_media_references_parentheses(self):
        assert _extract_media_references("![img](media/a.jpg)") == {"a.jpg"}
        assert _extract_media_references("![img](media/a(b).jpg)") == {"a(b).jpg"}
        assert _extract_media_references("![img](<media/a(b).jpg>)") == {"a(b).jpg"}

    def test_extract_media_references_url_encoding(self):
        assert _extract_media_references("![img](media/a%20b.jpg)") == {"a b.jpg"}
        assert _extract_media_references("![img](media/a%27s.jpg)") == {"a's.jpg"}

    def test_extract_media_references_html(self):
        assert _extract_media_references('<img src="a(b).jpg">') == {"a(b).jpg"}
        assert _extract_media_references('<img src="a b.jpg">') == {"a b.jpg"}
        assert _extract_media_references('<img src="media/a%20b.jpg">') == {"a b.jpg"}

    def test_extract_media_references_ignores_external(self):
        assert (
            _extract_media_references('<img src="http://example.com/a.jpg">') == set()
        )
        assert _extract_media_references("![img](https://example.com/a.jpg)") == set()

    def test_update_media_references_parentheses_and_html(self, fs, tmp_path):
        md_dir = tmp_path / "collection"
        md_dir.mkdir()
        md_file = md_dir / "deck.md"
        md_file.write_text(
            (
                "1. ![img](media/a(b).jpg)\n"
                '2. <img src="a b.jpg">\n'
                "3. [sound:a%20b.mp3]\n"
            ),
            encoding="utf-8",
        )

        rename_map = {
            "media/a(b).jpg": "media/hashed1.jpg",
            "a b.jpg": "media/hashed2.jpg",
            "a b.mp3": "hashed3.mp3",
        }
        updated_count = fs.update_media_references(md_dir, rename_map)
        assert updated_count == 1

        new_content = md_file.read_text(encoding="utf-8")
        assert "![img](<media/hashed1.jpg>)" in new_content
        assert '<img src="media/hashed2.jpg">' in new_content
        assert "[sound:hashed3.mp3]" in new_content


class TestMediaDirectory:
    """Local media directory scanning should be predictable."""

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


class TestMediaSyncIncremental:
    def test_sync_media_to_anki_warm_run_skips_unchanged_pushes(self, fs, tmp_path):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        (media_dir / "img.png").write_bytes(b"image-content")
        (tmp_path / "deck.md").write_text(
            "Q: Prompt\nA: ![img](media/img.png)", encoding="utf-8"
        )

        anki_media_dir = tmp_path / "anki_media"
        anki_media_dir.mkdir()
        anki = _FakeMediaAnki(anki_media_dir)

        db = SQLiteDbAdapter.load(tmp_path)
        try:
            first = sync_media_to_anki(anki, fs, tmp_path, db)
            second = sync_media_to_anki(anki, fs, tmp_path, db)
        finally:
            db.close()

        assert first.summary.synced == 1
        assert second.summary.synced == 0
        assert anki.push_count == 1

    def test_sync_media_to_anki_cache_persists_across_db_connections(
        self, fs, tmp_path
    ):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        (media_dir / "img.png").write_bytes(b"image-content")
        (tmp_path / "deck.md").write_text(
            "Q: Prompt\nA: ![img](media/img.png)", encoding="utf-8"
        )

        anki_media_dir = tmp_path / "anki_media"
        anki_media_dir.mkdir()
        anki = _FakeMediaAnki(anki_media_dir)

        db = SQLiteDbAdapter.load(tmp_path)
        try:
            first = sync_media_to_anki(anki, fs, tmp_path, db)
        finally:
            db.close()

        db = SQLiteDbAdapter.load(tmp_path)
        try:
            second = sync_media_to_anki(anki, fs, tmp_path, db)
        finally:
            db.close()

        assert first.summary.synced == 1
        assert second.summary.synced == 0
        assert anki.push_count == 1

    def test_sync_media_prunes_deleted_markdown_cache_rows(self, fs, tmp_path):
        media_dir = tmp_path / LOCAL_MEDIA_DIR
        media_dir.mkdir()
        (media_dir / "img.png").write_bytes(b"image-content")
        deck_a = tmp_path / "DeckA.md"
        deck_b = tmp_path / "DeckB.md"
        deck_a.write_text("Q: A\nA: ![img](media/img.png)", encoding="utf-8")
        deck_b.write_text("Q: B\nA: ![img](media/img.png)", encoding="utf-8")

        anki_media_dir = tmp_path / "anki_media"
        anki_media_dir.mkdir()
        anki = _FakeMediaAnki(anki_media_dir)

        db = SQLiteDbAdapter.load(tmp_path)
        try:
            sync_media_to_anki(anki, fs, tmp_path, db)
            assert db.get_markdown_media_cached_paths() == {"DeckA.md", "DeckB.md"}

            deck_a.unlink()
            sync_media_to_anki(anki, fs, tmp_path, db)
            assert db.get_markdown_media_cached_paths() == {"DeckB.md"}
        finally:
            db.close()

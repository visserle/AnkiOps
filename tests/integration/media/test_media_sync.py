"""Media synchronization behavior tests."""

from __future__ import annotations

import logging
from pathlib import Path

from ankiops.config import LOCAL_MEDIA_DIR
from ankiops.db import SQLiteDbAdapter
from ankiops.sync_media import sync_all_media_to_anki, sync_media_to_anki


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


def _sync_to_anki(fs, collection_dir: Path, anki: _FakeMediaAnki):
    db = SQLiteDbAdapter.open(collection_dir)
    try:
        return sync_media_to_anki(anki, fs, collection_dir, db)
    finally:
        db.close()


def test_sync_all_media_to_anki_resolves_media_relative_to_each_source(fs, tmp_path):
    root_media = tmp_path / LOCAL_MEDIA_DIR
    root_media.mkdir()
    (root_media / "image.png").write_bytes(b"root image")
    (tmp_path / "RootDeck.md").write_text(
        "Q: Root\nA: ![img](media/image.png)", encoding="utf-8"
    )

    shared_root = tmp_path / "shared" / "owner" / "repo"
    shared_media = shared_root / LOCAL_MEDIA_DIR
    shared_media.mkdir(parents=True)
    (shared_media / "image.png").write_bytes(b"shared image")
    (shared_root / "SharedDeck.md").write_text(
        "Q: Shared\nA: ![img](media/image.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        result = sync_all_media_to_anki(anki, fs, tmp_path, db)
    finally:
        db.close()

    root_names = sorted(path.name for path in root_media.glob("image_*.png"))
    shared_names = sorted(path.name for path in shared_media.glob("image_*.png"))
    assert len(root_names) == 1
    assert len(shared_names) == 1
    assert root_names != shared_names
    assert result.summary.synced == 2
    assert (anki_media_dir / root_names[0]).exists()
    assert (anki_media_dir / shared_names[0]).exists()
    assert f"media/{root_names[0]}" in (tmp_path / "RootDeck.md").read_text(
        encoding="utf-8"
    )
    assert f"media/{shared_names[0]}" in (shared_root / "SharedDeck.md").read_text(
        encoding="utf-8"
    )


def test_sync_media_to_anki_handles_markdown_html_audio_and_external_refs(fs, tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "a(b).jpg").write_bytes(b"paren image")
    (media_dir / "a b.jpg").write_bytes(b"space image")
    (media_dir / "clip.mp3").write_bytes(b"audio")
    (tmp_path / "Deck.md").write_text(
        (
            "Q: Media\n"
            "A: ![paren](<media/a(b).jpg>)\n"
            '<img src="media/a%20b.jpg">\n'
            "[sound:clip.mp3]\n"
            "![external](https://example.com/remote.png)\n"
        ),
        encoding="utf-8",
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    result = _sync_to_anki(fs, tmp_path, anki)

    content = (tmp_path / "Deck.md").read_text(encoding="utf-8")
    pushed_names = sorted(path.name for path in anki_media_dir.iterdir())
    assert result.checked == 3
    assert result.summary.synced == 3
    assert len(pushed_names) == 3
    assert "https://example.com/remote.png" in content
    assert "a(b)_" in content
    assert "a b_" in content
    assert "[sound:clip_" in content


def test_sync_media_to_anki_warm_run_skips_unchanged_pushes(fs, tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "img.png").write_bytes(b"image-content")
    (tmp_path / "deck.md").write_text(
        "Q: Prompt\nA: ![img](media/img.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        first = sync_media_to_anki(anki, fs, tmp_path, db)
        second = sync_media_to_anki(anki, fs, tmp_path, db)
    finally:
        db.close()

    assert first.summary.synced == 1
    assert second.summary.synced == 0
    assert anki.push_count == 1


def test_sync_media_to_anki_cache_persists_across_db_connections(fs, tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "img.png").write_bytes(b"image-content")
    (tmp_path / "deck.md").write_text(
        "Q: Prompt\nA: ![img](media/img.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        first = sync_media_to_anki(anki, fs, tmp_path, db)
    finally:
        db.close()

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        second = sync_media_to_anki(anki, fs, tmp_path, db)
    finally:
        db.close()

    assert first.summary.synced == 1
    assert second.summary.synced == 0
    assert anki.push_count == 1


def test_sync_media_prunes_deleted_markdown_cache_rows(fs, tmp_path):
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

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        sync_media_to_anki(anki, fs, tmp_path, db)
        assert db.list_markdown_media_paths() == {"DeckA.md", "DeckB.md"}

        deck_a.unlink()
        sync_media_to_anki(anki, fs, tmp_path, db)
        assert db.list_markdown_media_paths() == {"DeckB.md"}
    finally:
        db.close()


def test_sync_media_to_anki_reports_missing_local_references(fs, tmp_path, caplog):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "present.png").write_bytes(b"image-content")
    (tmp_path / "deck.md").write_text(
        (
            "Q: Prompt\n"
            "A: ![present](media/present.png)\n"
            "![missing](media/missing.png)\n"
        ),
        encoding="utf-8",
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SQLiteDbAdapter.open(tmp_path)
    try:
        with caplog.at_level(logging.WARNING):
            result = sync_media_to_anki(anki, fs, tmp_path, db)
    finally:
        db.close()

    assert result.checked == 2
    assert result.missing == 1
    assert result.summary.synced == 1
    assert anki.push_count == 1
    assert "missing.png referenced in markdown but missing in local media/" in (
        caplog.text
    )

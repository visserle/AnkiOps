"""Media synchronization behavior tests."""

from __future__ import annotations

from pathlib import Path

from ankiops.collection import LOCAL_MEDIA_DIR
from ankiops.deck_sources import DeckSource
from ankiops.git import GitRepository
from ankiops.media import (
    sync_all_media_to_anki,
    sync_media_to_anki,
)
from ankiops.sync.state import SyncState


class _FakeMediaAnki:
    def __init__(self, media_dir: Path):
        self._media_dir = media_dir
        self.push_count = 0
        self.pull_count = 0

    def get_media_dir(self) -> Path:
        return self._media_dir

    def push_media(self, local_path: Path, remote_filename: str) -> None:
        self.push_count += 1
        target = self._media_dir / remote_filename
        target.write_bytes(local_path.read_bytes())

    def pull_media(self, remote_filename: str, local_path: Path) -> bool:
        source = self._media_dir / remote_filename
        if not source.exists():
            return False
        self.pull_count += 1
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(source.read_bytes())
        return True

    def delete_media_file(self, remote_filename: str) -> None:
        (self._media_dir / remote_filename).unlink(missing_ok=True)


def _sync_to_anki(collection_root: Path, anki: _FakeMediaAnki):
    db = SyncState.open(collection_root)
    try:
        return sync_media_to_anki(anki, DeckSource.local(collection_root), db)
    finally:
        db.close()


def test_sync_all_media_to_anki_resolves_media_relative_to_each_source(tmp_path):
    root_media = tmp_path / LOCAL_MEDIA_DIR
    root_media.mkdir()
    (root_media / "image.png").write_bytes(b"root image")
    (tmp_path / "RootDeck.md").write_text(
        "Q: Root\nA: ![img](media/image.png)", encoding="utf-8"
    )

    collab_root = tmp_path / "collab" / "owner" / "repo"
    collab_media = collab_root / LOCAL_MEDIA_DIR
    collab_media.mkdir(parents=True)
    GitRepository(collab_root).init_repo()
    (collab_media / "image.png").write_bytes(b"collab image")
    (collab_root / "CollabDeck.md").write_text(
        "Q: Collab\nA: ![img](media/image.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SyncState.open(tmp_path)
    try:
        result = sync_all_media_to_anki(anki, tmp_path, db)
    finally:
        db.close()

    root_names = sorted(path.name for path in root_media.glob("image_*.png"))
    collab_names = sorted(path.name for path in collab_media.glob("image_*.png"))
    assert len(root_names) == 1
    assert len(collab_names) == 1
    assert root_names != collab_names
    assert result.summary.synced == 2
    assert (anki_media_dir / root_names[0]).exists()
    assert (anki_media_dir / collab_names[0]).exists()
    assert f"media/{root_names[0]}" in (tmp_path / "RootDeck.md").read_text(
        encoding="utf-8"
    )
    assert f"media/{collab_names[0]}" in (collab_root / "CollabDeck.md").read_text(
        encoding="utf-8"
    )


def test_sync_all_media_deletes_unreferenced_managed_anki_media(tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "remove.png").write_bytes(b"managed image")
    deck = tmp_path / "Deck.md"
    deck.write_text("Q: media\nA: ![remove](media/remove.png)\n", encoding="utf-8")
    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)
    db = SyncState.open(tmp_path)
    try:
        sync_all_media_to_anki(anki, tmp_path, db)
        managed_name = next(path.name for path in anki_media_dir.iterdir())
        deck.write_text("Q: media\nA: removed\n", encoding="utf-8")

        result = sync_all_media_to_anki(anki, tmp_path, db)
    finally:
        db.close()

    assert not (anki_media_dir / managed_name).exists()
    assert result.summary.deleted == 1


def test_sync_media_to_anki_does_not_rewrite_readme_references(tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "image.png").write_bytes(b"deck image")
    (tmp_path / "Deck.md").write_text(
        "Q: Deck\nA: ![img](media/image.png)", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "# Docs\n\n![img](media/image.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    result = _sync_to_anki(tmp_path, anki)

    deck_content = (tmp_path / "Deck.md").read_text(encoding="utf-8")
    readme_content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert result.checked == 1
    assert "image_" in deck_content
    assert "media/image.png" in readme_content


def test_sync_media_to_anki_handles_markdown_html_audio_and_external_refs(tmp_path):
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

    result = _sync_to_anki(tmp_path, anki)

    content = (tmp_path / "Deck.md").read_text(encoding="utf-8")
    pushed_names = sorted(path.name for path in anki_media_dir.iterdir())
    assert result.checked == 3
    assert result.summary.synced == 3
    assert len(pushed_names) == 3
    assert "https://example.com/remote.png" in content
    assert "a(b)_" in content
    assert "a b_" in content
    assert "[sound:clip_" in content


def test_sync_media_to_anki_cache_persists_across_db_connections(tmp_path):
    media_dir = tmp_path / LOCAL_MEDIA_DIR
    media_dir.mkdir()
    (media_dir / "img.png").write_bytes(b"image-content")
    (tmp_path / "deck.md").write_text(
        "Q: Prompt\nA: ![img](media/img.png)", encoding="utf-8"
    )

    anki_media_dir = tmp_path / "anki_media"
    anki_media_dir.mkdir()
    anki = _FakeMediaAnki(anki_media_dir)

    db = SyncState.open(tmp_path)
    try:
        first = sync_media_to_anki(anki, DeckSource.local(tmp_path), db)
    finally:
        db.close()

    db = SyncState.open(tmp_path)
    try:
        second = sync_media_to_anki(anki, DeckSource.local(tmp_path), db)
    finally:
        db.close()

    assert first.summary.synced == 1
    assert second.summary.synced == 0
    assert anki.push_count == 1


def test_sync_media_prunes_deleted_markdown_cache_rows(tmp_path):
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

    db = SyncState.open(tmp_path)
    try:
        sync_all_media_to_anki(anki, tmp_path, db)
        assert db.list_markdown_media_paths() == {"DeckA.md", "DeckB.md"}

        deck_a.unlink()
        sync_all_media_to_anki(anki, tmp_path, db)
        assert db.list_markdown_media_paths() == {"DeckB.md"}
    finally:
        db.close()


def test_sync_media_to_anki_reports_missing_local_references(tmp_path):
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

    db = SyncState.open(tmp_path)
    try:
        result = sync_media_to_anki(anki, DeckSource.local(tmp_path), db)
    finally:
        db.close()

    assert result.checked == 2
    assert result.missing == 1
    assert result.summary.synced == 1
    assert anki.push_count == 1

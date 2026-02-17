"""Tests for media synchronization logic."""

import hashlib
from pathlib import Path

from ankiops.sync_media import (
    _calculate_hash,
    _get_hashed_filename,
    apply_hashing,
    sync_from_anki,
    sync_to_anki,
    update_markdown_media_references,
    LOCAL_MEDIA_DIR,
)


def create_image(path: Path, content: bytes = b"fake image content"):
    """Helper to create a dummy image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_calculate_hash(tmp_path):
    """Test SHA256 hash calculation."""
    f = tmp_path / "test.png"
    create_image(f, b"test content")
    
    expected_hash = hashlib.sha256(b"test content").hexdigest()[:8]
    assert _calculate_hash(f) == expected_hash


def test_get_hashed_filename():
    """Test filename generation with hash."""
    path = Path("media/image.png")
    h = "a1b2c3d4"
    assert _get_hashed_filename(path, h) == "image_a1b2c3d4.png"

    # Idempotency check
    path_hashed = Path("media/image_a1b2c3d4.png")
    assert _get_hashed_filename(path_hashed, h) == "image_a1b2c3d4.png"

    # Re-hashing (content changed)
    new_h = "e5f6g7h8"
    assert _get_hashed_filename(path_hashed, new_h) == "image_e5f6g7h8.png"


def test_update_markdown_media_references(tmp_path):
    """Test updating markdown references."""
    collection_dir = tmp_path
    md_file = collection_dir / "test.md"
    md_file.write_text(
        "![img](media/old.png)\n"
        "<img src=\"media/old.png\">\n"
        "[sound:media/audio.mp3]"
    )

    rename_map = {
        "media/old.png": "media/new.png",
        "media/audio.mp3": "media/audio_hash.mp3",
    }

    update_markdown_media_references(collection_dir, rename_map)

    content = md_file.read_text()
    assert "![img](media/new.png)" in content
    assert '<img src="media/new.png">' in content
    assert "[sound:media/audio_hash.mp3]" in content


def test_apply_hashing(tmp_path):
    """Test hashing and renaming of local files."""
    collection_dir = tmp_path
    media_dir = collection_dir / LOCAL_MEDIA_DIR
    media_dir.mkdir()

    # Create files
    img1 = media_dir / "image1.png"
    create_image(img1, b"content1")
    
    # Check that subfolders are IGNORED
    (media_dir / "sub").mkdir()
    img2 = media_dir / "sub" / "image2.png"
    create_image(img2, b"content2")

    # Create markdown referencing them
    md_file = collection_dir / "deck.md"
    md_file.write_text("![1](media/image1.png)")

    apply_hashing(collection_dir)

    # Calculate expected hashes
    h1 = hashlib.sha256(b"content1").hexdigest()[:8]

    # Verify renames
    new_img1 = media_dir / f"image1_{h1}.png"

    assert not img1.exists()
    assert new_img1.exists()
    
    # Verify subfolder file was NOT touched (ignored)
    assert img2.exists()
    assert not (media_dir / "sub" / f"image2_{hashlib.sha256(b'content2').hexdigest()[:8]}.png").exists()

    # Verify markdown update
    content = md_file.read_text()
    assert f"media/image1_{h1}.png" in content


def test_apply_hashing_idempotent(tmp_path):
    """Test that applying hashing to already correct files does nothing."""
    collection_dir = tmp_path
    media_dir = collection_dir / LOCAL_MEDIA_DIR
    media_dir.mkdir()

    content = b"content"
    h = hashlib.sha256(content).hexdigest()[:8]
    fname = f"image_{h}.png"
    
    img = media_dir / fname
    create_image(img, content)

    # Run hashing
    apply_hashing(collection_dir)

    # Verify no change (file still exists, no double hash)
    assert img.exists()
    assert not (media_dir / f"image_{h}_{h}.png").exists()


def test_sync_to_anki(tmp_path):
    """Test syncing files to Anki directory."""
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"
    
    collection_dir.mkdir()
    anki_media.mkdir()
    (collection_dir / LOCAL_MEDIA_DIR).mkdir()

    # Setup local file
    img = collection_dir / LOCAL_MEDIA_DIR / "test.png"
    create_image(img, b"data")
    
    # Run sync
    sync_to_anki(collection_dir, anki_media)
    
    # Expect file to be hashed and copied
    h = hashlib.sha256(b"data").hexdigest()[:8]
    expected_name = f"test_{h}.png"
    
    assert (anki_media / expected_name).exists()
    assert (anki_media / expected_name).read_bytes() == b"data"


def test_sync_from_anki(tmp_path):
    """Test syncing referenced files from Anki."""
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"
    
    collection_dir.mkdir()
    anki_media.mkdir()

    # Setup Anki file
    (anki_media / "remote.png").write_bytes(b"remote data")

    # References
    refs = {"remote.png", "missing.png"}

    sync_from_anki(collection_dir, anki_media, refs)

    media_dir = collection_dir / LOCAL_MEDIA_DIR
    assert (media_dir / "remote.png").exists()
    assert (media_dir / "remote.png").read_bytes() == b"remote data"
    assert not (media_dir / "missing.png").exists()

import hashlib
from pathlib import Path

import pytest

from ankiops.sync_media import (
    LOCAL_MEDIA_DIR,
    _calculate_hash,
    _get_hashed_filename,
    apply_hashing,
    extract_media_references,
    sync_from_anki,
    sync_to_anki,
    update_markdown_media_references,
)


def create_image(path: Path, content: bytes = b"fake image content"):
    """Helper to create a dummy image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_sync_to_anki_safety_aliased_dirs(tmp_path):
    """Test that sync_to_anki raises error if local media dir IS the anki media dir."""
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"

    collection_dir.mkdir()
    anki_media.mkdir()
    (collection_dir / "media").symlink_to(anki_media)

    # Create a file in Anki media
    original_file = anki_media / "pre_existing.png"
    create_image(original_file, b"content")

    # Assert that safety check triggers
    with pytest.raises(ValueError, match="aliases Anki media directory"):
        sync_to_anki(collection_dir, anki_media)

    # Verify file was NOT touched/renamed
    assert original_file.exists()
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"content")
    h = digest.hexdigest()
    assert not (anki_media / f"pre_existing_{h}.png").exists()


def test_calculate_hash(tmp_path):
    """Test BLAKE2b hash calculation."""
    f = tmp_path / "test.png"
    create_image(f, b"test content")

    # Calculate expected hash using same algo as implementation
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"test content")
    expected_hash = digest.hexdigest()

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
        '![img](media/old.png)\n<img src="media/old.png">\n[sound:media/audio.mp3]'
    )

    rename_map = {
        "media/old.png": "media/new.png",
        "media/audio.mp3": "media/audio_hash.mp3",
    }

    update_markdown_media_references(collection_dir, rename_map)

    content = md_file.read_text()
    assert "![img](<media/new.png>)" in content
    assert (
        '<img src="<media/new.png>">' in content
        or '<img src="<media/new.png>">' in content
    )
    assert "[sound:<media/audio_hash.mp3>]" in content


def test_update_media_references_with_angle_brackets(tmp_path):
    """Test updating markdown references that use angle brackets."""
    collection_dir = tmp_path
    md_file = collection_dir / "test_brackets.md"
    md_file.write_text(
        "![img](<media/old.png>)\n<img src='<media/old.png>'>\n[sound:<media/audio.mp3>]"
    )

    rename_map = {
        "media/old.png": "media/new.png",
        "media/audio.mp3": "media/audio_hash.mp3",
    }

    update_markdown_media_references(collection_dir, rename_map)

    content = md_file.read_text()
    assert "![img](<media/new.png>)" in content
    # Angle brackets should be preserved (or added if logic dictates, but main goal is proper replacement)
    # The fix ensures brackets are preserved if present in original match
    assert "src='<media/new.png>'" in content
    assert "[sound:<media/audio_hash.mp3>]" in content


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

    # Create underscore file (should be ignored)
    underscore = media_dir / "_ignore.png"
    create_image(underscore, b"ignore")

    apply_hashing(collection_dir)

    # Calculate expected hashes
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"content1")
    h1 = digest.hexdigest()

    # Verify renames
    new_img1 = media_dir / f"image1_{h1}.png"

    assert not img1.exists()
    assert new_img1.exists()

    # Verify underscore file was NOT touched
    assert underscore.exists()
    assert not (media_dir / f"_ignore_{h1}.png").exists()

    # Verify subfolder file was NOT touched (ignored)
    assert img2.exists()

    # Calculate hash for second file for negative check
    digest2 = hashlib.blake2b(digest_size=4)
    digest2.update(b"content2")
    h2 = digest2.hexdigest()

    assert not (media_dir / "sub" / f"image2_{h2}.png").exists()

    # Verify markdown update
    content = md_file.read_text()
    assert f"media/image1_{h1}.png" in content
    # Note: sync_media.py uses replace(path, new_path)
    # If the original was ![1](media/image1.png), it replaces "media/image1.png" with "<media/image1_hash.png>"
    # Result: ![1](<media/image1_hash.png>)
    assert f"![1](<media/image1_{h1}.png>)" in content


def test_apply_hashing_idempotent(tmp_path):
    """Test that applying hashing to already correct files does nothing."""
    collection_dir = tmp_path
    media_dir = collection_dir / LOCAL_MEDIA_DIR
    media_dir.mkdir()

    content = b"content"

    digest = hashlib.blake2b(digest_size=4)
    digest.update(content)
    h = digest.hexdigest()

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

    # Create markdown referencing it
    (collection_dir / "deck.md").write_text("![img](media/test.png)")

    # Run sync
    summary = sync_to_anki(collection_dir, anki_media)
    assert summary.summary.synced == 1
    assert summary.summary.total == 1

    # Expect file to be hashed and copied
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"data")
    h = digest.hexdigest()

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

    summary = sync_from_anki(collection_dir, anki_media, refs)
    assert summary.summary.synced == 1
    assert summary.summary.total == 2

    media_dir = collection_dir / LOCAL_MEDIA_DIR
    assert (media_dir / "remote.png").exists()
    assert (media_dir / "remote.png").read_bytes() == b"remote data"
    assert not (media_dir / "missing.png").exists()


def test_extract_markdown_images():
    """Test extracting markdown image references."""
    text = "Some text ![alt](image.png) more text ![](another.jpg)"
    refs = extract_media_references(text)
    assert "image.png" in refs
    assert "another.jpg" in refs


def test_extract_media_prefix():
    """Test extracting references with media/ prefix."""
    text = "![](media/image.png) and [sound:media/audio.mp3]"
    refs = extract_media_references(text)
    # Should strip media/ prefix
    assert "image.png" in refs
    assert "audio.mp3" in refs
    assert "media/image.png" not in refs


def test_extract_sound_tags():
    """Test extracting Anki sound tags."""
    text = "Text with [sound:audio.mp3] and [sound:music.ogg]"
    refs = extract_media_references(text)
    assert "audio.mp3" in refs
    assert "music.ogg" in refs


def test_extract_html_img():
    """Test extracting HTML img tags."""
    text = '<img src="image.png"> and <img src="photo.jpg">'
    refs = extract_media_references(text)
    assert "image.png" in refs
    assert "photo.jpg" in refs


def test_sync_to_anki_with_cleanup(tmp_path):
    """Test that sync_to_anki syncs referenced files and cleans up unreferenced ones."""
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"

    collection_dir.mkdir()
    anki_media.mkdir()
    (collection_dir / LOCAL_MEDIA_DIR).mkdir()

    # 1. Setup local files
    img1 = collection_dir / LOCAL_MEDIA_DIR / "referenced.png"
    create_image(img1, b"referenced data")

    img2 = collection_dir / LOCAL_MEDIA_DIR / "unreferenced.png"
    create_image(img2, b"unreferenced data")

    img3 = collection_dir / LOCAL_MEDIA_DIR / "_static.png"
    create_image(img3, b"static data")

    # 2. Setup markdown referencing only one image
    md_file = collection_dir / "deck.md"
    md_file.write_text("![img](media/referenced.png)")

    # 3. Create expected names (hashing will happen inside sync_to_anki)
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"referenced data")
    h = digest.hexdigest()
    expected_referenced_name = f"referenced_{h}.png"

    # 4. Run sync
    summary = sync_to_anki(collection_dir, anki_media)
    assert summary.summary.synced == 2
    assert summary.summary.deleted == 1
    assert summary.summary.total == 2  # referenced (hashed) and _static

    # 5. Assertions

    # Referenced file: Should be in Anki (hashed) AND Local (hashed)
    assert (anki_media / expected_referenced_name).exists()
    assert (collection_dir / LOCAL_MEDIA_DIR / expected_referenced_name).exists()

    # Unreferenced file: Should NOT be in Anki AND should be DELETED from Local
    # Note: Using glob because hashing might rename it if logic is broken;
    # but correct logic means it's gone regardless of name.
    assert not list(anki_media.glob("unreferenced*"))
    assert not list((collection_dir / LOCAL_MEDIA_DIR).glob("unreferenced*"))

    # Underscore file: Should be in Anki (synced) AND should REMAIN in Local
    assert (anki_media / "_static.png").exists()
    assert (collection_dir / LOCAL_MEDIA_DIR / "_static.png").exists()


def test_sync_to_anki_syncs_underscores(tmp_path):
    """Test that underscore files are synced to Anki even if not referenced in markdown."""
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"

    collection_dir.mkdir()
    anki_media.mkdir()
    (collection_dir / LOCAL_MEDIA_DIR).mkdir()

    # Setup local underscore file
    underscore_img = collection_dir / LOCAL_MEDIA_DIR / "_static.png"
    create_image(underscore_img, b"static data")

    # Run sync (no markdown references created)
    summary = sync_to_anki(collection_dir, anki_media)
    assert summary.summary.synced == 1
    assert summary.summary.total == 1

    # Assert underscore file is copied to Anki
    assert (anki_media / "_static.png").exists()
    assert (anki_media / "_static.png").read_bytes() == b"static data"


def test_roundtrip_import_local_media(tmp_path):
    """Test standardizing a local unhashed image reference during import.

    Scenario:
    1. Local has `media/image.png`.
    2. Markdown has `![alt](image.png)` (no prefix).
    3. Sync to Anki.
    4. Expect:
       - File hashed to `media/image_<hash>.png`
       - Markdown updated to `![alt](media/image_<hash>.png)`
       - File synced to Anki
    """
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"

    collection_dir.mkdir()
    anki_media.mkdir()
    (collection_dir / LOCAL_MEDIA_DIR).mkdir()

    # 1. Setup local file
    img_name = "image.png"
    img_path = collection_dir / LOCAL_MEDIA_DIR / img_name
    create_image(img_path, b"image data")

    # 2. Setup markdown referencing it without prefix
    md_file = collection_dir / "deck.md"
    md_file.write_text(f"![alt]({img_name})")

    # 3. Run sync
    sync_to_anki(collection_dir, anki_media)

    # 4. Verify
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"image data")
    h = digest.hexdigest()
    expected_name = f"image_{h}.png"

    # Markdown updated
    content = md_file.read_text()
    assert f"<media/{expected_name}>" in content

    # File exists locally (hashed)
    assert (collection_dir / LOCAL_MEDIA_DIR / expected_name).exists()
    assert not (collection_dir / LOCAL_MEDIA_DIR / img_name).exists()

    # File exists in Anki
    assert (anki_media / expected_name).exists()


def test_roundtrip_export_remote_media(tmp_path):
    """Test downloading and then standardizing a remote image reference.

    Scenario:
    1. Anki has `remote.png` (unhashed).
    2. Markdown has `![alt](remote.png)` (simulating raw export from Anki).
    3. Sync From Anki (download).
    4. Sync To Anki (import/standardize).
    5. Expect:
       - File downloaded to local
       - File hashed and markdown updated
       - Hashed file returned to Anki
    """
    collection_dir = tmp_path / "collection"
    anki_media = tmp_path / "anki_media"

    collection_dir.mkdir()
    anki_media.mkdir()

    # 1. Setup Anki file
    remote_img = anki_media / "remote.png"
    create_image(remote_img, b"remote data")

    # 2. Setup markdown (raw reference)
    md_file = collection_dir / "deck.md"
    md_file.write_text("![alt](remote.png)")

    # 3. Sync From Anki
    summary = sync_from_anki(collection_dir, anki_media, {"remote.png"})
    assert summary.summary.synced == 1

    # Verify download
    local_img = collection_dir / LOCAL_MEDIA_DIR / "remote.png"
    assert local_img.exists()
    assert local_img.read_bytes() == b"remote data"

    # 4. Sync To Anki
    sync_to_anki(collection_dir, anki_media)

    # 5. Verify standardization
    digest = hashlib.blake2b(digest_size=4)
    digest.update(b"remote data")
    h = digest.hexdigest()
    expected_name = f"remote_{h}.png"

    # Markdown updated
    content = md_file.read_text()
    assert f"<media/{expected_name}>" in content

    # Local file hashed
    assert (collection_dir / LOCAL_MEDIA_DIR / expected_name).exists()
    assert not local_img.exists()

    # Anki has new file (it keeps old one too usually, but check new one exists)
    assert (anki_media / expected_name).exists()

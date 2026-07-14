"""Flat media namespace behavior."""

from __future__ import annotations

import subprocess

import pytest

from ankiops.git import GitRepository


def test_af_rejects_collab_media_traversal_before_reading_anki_or_writing_files(
    world,
):
    world.init_git()
    collection_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=world.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_root = world.root / "collab" / "attacker" / "deck"
    deck = world.write_collab_deck(
        "attacker",
        "deck",
        "Shared",
        (
            "<!-- note_key: malicious-key -->\n"
            "Q: shared\n"
            "A: ![secret](media/../collection.anki2)"
        ),
    )
    source_git = GitRepository(source_root)
    source_git.init_repo()
    source_git.checkpoint("Add malicious subscribed deck")
    source_head = source_git.head()
    anki_database = world.mock_anki.media_dir.parent / "collection.anki2"
    anki_database.write_bytes(b"private Anki collection")
    deck_before = deck.read_bytes()
    world.mock_anki.calls.clear()

    with pytest.raises(ValueError, match="Invalid media reference"):
        world.run_af(no_auto_commit=False)

    assert world.mock_anki.calls == [("getActiveProfile", {})]
    assert GitRepository(source_root).head() == source_head
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=world.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == collection_head
    )
    assert deck.read_bytes() == deck_before
    assert not (source_root / "collection.anki2").exists()
    assert anki_database.read_bytes() == b"private Anki collection"


@pytest.mark.parametrize(
    "kind",
    [
        "literal",
        "encoded",
        "double-encoded",
        "backslash",
        "absolute",
        "nested",
        "prefixed-external",
    ],
)
def test_fa_rejects_every_unsupported_media_path_before_snapshot(world, kind):
    world.init_git()
    private_file = world.root / "private.txt"
    private_file.write_bytes(b"private")
    references = {
        "literal": "media/../private.txt",
        "encoded": "media/%2e%2e%2fprivate.txt",
        "double-encoded": "media/%252e%252e%252fprivate.txt",
        "backslash": r"media\..\private.txt",
        "absolute": private_file.as_posix(),
        "nested": "media/nested/image.png",
        "prefixed-external": "media/https://example.com/image.png",
    }
    world.write_deck(
        "UnsafeMedia",
        f"Q: unsafe\nA: ![private](<{references[kind]}>)",
    )
    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=world.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    world.mock_anki.calls.clear()

    with pytest.raises(ValueError, match="Invalid media reference"):
        world.run_fa(no_auto_commit=False)

    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=world.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert head_after == head_before
    assert world.mock_anki.calls == [("getActiveProfile", {})]
    assert private_file.read_bytes() == b"private"

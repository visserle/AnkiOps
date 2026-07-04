from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from ankiops.collab.commands import run_update
from ankiops.sync.state import SyncState
from tests.support.deck_files import DeckFileHarness


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _identity(root: Path, name: str) -> None:
    _git(root, "config", "user.name", name)
    _git(root, "config", "user.email", f"{name}@example.test")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def test_concurrent_collab_update_preserves_private_root(tmp_path, monkeypatch):
    collection = tmp_path / "collection"
    collection.mkdir()
    _git(collection, "init", "-b", "main")
    _identity(collection, "private")
    (collection / ".gitignore").write_text(
        "/collab/\n.ankiops.db*\n.ankiops/\n", encoding="utf-8"
    )
    private = collection / "Private.md"
    private.write_text("private baseline\n", encoding="utf-8")
    _commit(collection, "Initialize private collection")
    db = SyncState.open(collection)
    db.close()

    remote = tmp_path / "source.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    upstream = tmp_path / "upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _identity(upstream, "upstream")
    DeckFileHarness().eject_default_note_types(upstream / "note_types")
    (upstream / "Deck.md").write_text(
        "<!-- note_key: collab -->\nQ: question\nA: baseline\n",
        encoding="utf-8",
    )
    _commit(upstream, "Initial collab deck")
    _git(upstream, "push", "origin", "main")

    source = collection / "collab" / "owner" / "repo"
    source.parent.mkdir(parents=True)
    _git(source.parent, "clone", "--origin", "upstream", str(remote), str(source))
    _identity(source, "local")
    _git(source, "checkout", "-b", "ankiops/work", "upstream/main")

    (source / "Local.md").write_text(
        "<!-- note_key: local -->\nQ: local\nA: local\n", encoding="utf-8"
    )
    private.write_text("private dirty\n", encoding="utf-8")
    (upstream / "Deck.md").write_text(
        (upstream / "Deck.md")
        .read_text(encoding="utf-8")
        .replace("baseline", "remote edit"),
        encoding="utf-8",
    )
    _commit(upstream, "Remote collab edit")
    _git(upstream, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "remote edit" in (source / "Deck.md").read_text(encoding="utf-8")
    assert (source / "Local.md").exists()
    assert private.read_text(encoding="utf-8") == "private dirty\n"
    assert _git(collection, "status", "--short") == "M Private.md"

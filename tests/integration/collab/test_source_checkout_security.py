from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.collab.commands import run_subscribe, run_update
from ankiops.collab.git_state import INTEGRATED_REF
from ankiops.git import GitRepository
from ankiops.sync.state import SyncState
from tests.support.deck_files import DeckFileHarness


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def _configure(root: Path, name: str) -> None:
    _git(root, "config", "user.name", name)
    _git(root, "config", "user.email", f"{name.replace(' ', '.')}@example.test")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _collection(tmp_path: Path) -> Path:
    collection = tmp_path / "collection"
    collection.mkdir()
    _git(collection, "init", "-b", "main")
    _configure(collection, "Private User")
    (collection / ".gitignore").write_text(
        "/collab/\n.ankiops.db\n.ankiops.db-shm\n.ankiops.db-wal\n.ankiops/\n",
        encoding="utf-8",
    )
    (collection / "Private.md").write_text("private baseline\n", encoding="utf-8")
    _commit(collection, "Initialize private collection")
    state = SyncState.open(collection)
    state.set_profile_name("Test")
    state.close()
    return collection


def _safe_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    (seed / "Deck.md").write_text(
        "<!-- note_key: collab-key -->\nQ: shared\nA: safe answer\n",
        encoding="utf-8",
    )
    _commit(seed, "Add safe deck")
    _git(seed, "push", "origin", "main")
    return remote


def _checkout_source(collection: Path, remote: Path) -> Path:
    source = collection / "collab" / "owner" / "repo"
    source.parent.mkdir(parents=True)
    _git(source.parent, "clone", "--origin", "upstream", str(remote), str(source))
    _configure(source, "Contributor")
    _git(source, "checkout", "-b", "ankiops/journal", "upstream/main")
    _git(source, "branch", "--unset-upstream")
    _git(source, "update-ref", INTEGRATED_REF, "upstream/main")
    return source


def _redirect_clone(monkeypatch, upstream: Path) -> None:
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        return original_clone(
            cls,
            str(upstream),
            target,
            remote=remote,
            anonymous=anonymous,
        )

    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))


def test_subscribe_rejects_a_tracked_deck_symlink_before_reading_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    attacker = tmp_path / "attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "Deck.md").unlink()
    (attacker / "Deck.md").symlink_to("../../../Private.md")
    _commit(attacker, "Replace deck with symlink")
    _git(attacker, "push", "origin", "main")
    _redirect_clone(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert not (collection / "collab" / "owner" / "repo").exists()
    assert (collection / "Private.md").read_text() == "private baseline\n"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "note_types/AnkiOpsQA/note_type.yaml",
        "note_types/AnkiOpsQA/Front.template.anki",
        "note_types/AnkiOpsStyling.css",
        "note_types/AnkiOpsQA",
    ],
)
def test_subscribe_rejects_symlinked_note_type_inputs(
    tmp_path: Path, monkeypatch, unsafe_path: str
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    attacker = tmp_path / "attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    unsafe = attacker / unsafe_path
    if unsafe.is_dir():
        shutil.rmtree(unsafe)
    else:
        unsafe.unlink()
    unsafe.symlink_to("/private/ankiops-secret")
    _commit(attacker, "Replace note type input with symlink")
    _git(attacker, "push", "origin", "main")
    _redirect_clone(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert not (collection / "collab" / "owner" / "repo").exists()


def test_update_rejects_an_upstream_deck_symlink_before_applying_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    before = _git(source, "rev-parse", "HEAD").stdout.strip()
    attacker = tmp_path / "attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "Deck.md").unlink()
    (attacker / "Deck.md").symlink_to("../../../Private.md")
    _commit(attacker, "Replace deck with symlink")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert not (source / "Deck.md").is_symlink()
    assert (collection / "Private.md").read_text() == "private baseline\n"


def test_update_rejects_an_upstream_note_type_path_escape_before_applying_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    before = _git(source, "rev-parse", "HEAD").stdout.strip()
    attacker = tmp_path / "attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    manifest = attacker / "note_types" / "AnkiOpsQA" / "note_type.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "styling:\n  - ../AnkiOpsStyling.css\n  - ../SyntaxHighlighting.css",
            "styling: ../../../../../Private.md",
        ),
        encoding="utf-8",
    )
    _commit(attacker, "Escape note type directory")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="invalid styling file"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(source, "status", "--short").stdout == ""
    assert (collection / "Private.md").read_text() == "private baseline\n"

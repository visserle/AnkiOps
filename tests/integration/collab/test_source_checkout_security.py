from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.collab.commands import run_subscribe, run_update
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


def _malicious_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    (seed / "Deck.md").symlink_to("../../../Private.md")
    _commit(seed, "Add escaping deck symlink")
    _git(seed, "push", "origin", "main")
    return remote


def _manifest_escape_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "manifest-upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "manifest-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    manifest = seed / "note_types" / "AnkiOpsQA" / "note_type.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "styling:\n  - ../AnkiOpsStyling.css\n  - ../SyntaxHighlighting.css",
            "styling: ../../../../../Private.md",
        ),
        encoding="utf-8",
    )
    (seed / "Deck.md").write_text(
        "<!-- note_key: collab-key -->\nQ: shared\nA: answer\n",
        encoding="utf-8",
    )
    _commit(seed, "Reference a private file from the note-type manifest")
    _git(seed, "push", "origin", "main")
    return remote


def _safe_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "safe-upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "safe-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    (seed / ".gitignore").write_text("private.tmp\n", encoding="utf-8")
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
    _git(source, "update-ref", "refs/ankiops/integrated", "upstream/main")
    return source


def test_subscribe_rejects_a_tracked_deck_symlink_before_reading_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _malicious_remote(tmp_path)
    private = collection / "Private.md"
    private_before = private.read_bytes()
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        return original_clone(
            cls,
            str(upstream),
            target,
            remote=remote,
            anonymous=anonymous,
        )

    upstream = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert (
        _git(remote, "ls-tree", "main", "Deck.md").stdout.split(maxsplit=1)[0]
        == "120000"
    )
    assert private.read_bytes() == private_before
    assert not (collection / "collab" / "owner" / "repo").exists()


def test_subscribe_rejects_a_note_type_manifest_path_outside_the_repository(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _manifest_escape_remote(tmp_path)
    private = collection / "Private.md"
    private_before = private.read_bytes()
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        return original_clone(
            cls,
            str(upstream),
            target,
            remote=remote,
            anonymous=anonymous,
        )

    upstream = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="invalid styling file"):
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert private.read_bytes() == private_before
    assert not (collection / "collab" / "owner" / "repo").exists()


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
    attacker = tmp_path / "note-type-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    unsafe = attacker / unsafe_path
    if unsafe.is_dir():
        shutil.rmtree(unsafe)
    else:
        unsafe.unlink()
    unsafe.symlink_to("/private/ankiops-secret")
    _commit(attacker, "Replace note-type input with symlink")
    _git(attacker, "push", "origin", "main")
    assert _git(remote, "ls-tree", "main", unsafe_path).stdout.startswith("120000 ")
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        return original_clone(
            cls,
            str(upstream),
            target,
            remote=remote,
            anonymous=anonymous,
        )

    upstream = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
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
    private = collection / "Private.md"
    deck = source / "Deck.md"
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    deck_before = deck.read_bytes()
    private_before = private.read_bytes()

    attacker = tmp_path / "attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "Deck.md").unlink()
    (attacker / "Deck.md").symlink_to("../../../Private.md")
    _commit(attacker, "Replace deck with escaping symlink")
    _git(attacker, "push", "origin", "main")
    assert (
        _git(remote, "ls-tree", "main", "Deck.md").stdout.split(maxsplit=1)[0]
        == "120000"
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert not deck.is_symlink()
    assert deck.read_bytes() == deck_before
    assert private.read_bytes() == private_before


def test_update_rejects_an_upstream_note_type_path_escape_before_applying_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = collection / "Private.md"
    deck = source / "Deck.md"
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    deck_before = deck.read_bytes()
    private_before = private.read_bytes()

    attacker = tmp_path / "manifest-update-attacker"
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
    _commit(attacker, "Escape the note-types directory")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="invalid styling file"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert deck.read_bytes() == deck_before
    assert private.read_bytes() == private_before


def test_update_rejects_a_symlinked_local_deck_before_reading_it(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = collection / "Private.md"
    deck = source / "Deck.md"
    deck.unlink()
    deck.symlink_to("../../../Private.md")
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    private_before = private.read_bytes()
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert deck.is_symlink()
    assert private.read_bytes() == private_before


@pytest.mark.parametrize(
    "upstream_content",
    ["upstream public bytes\n", "local private bytes\n"],
    ids=["different-bytes", "identical-bytes"],
)
def test_update_does_not_overwrite_an_ignored_file_newly_tracked_upstream(
    tmp_path: Path, monkeypatch, upstream_content
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = source / "private.tmp"
    private.write_text("local private bytes\n", encoding="utf-8")
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    private_before = private.read_bytes()

    attacker = tmp_path / "ignored-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "private.tmp").write_text(upstream_content, encoding="utf-8")
    _git(attacker, "add", "-f", "private.tmp")
    _git(attacker, "commit", "-m", "Track formerly private path")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="untracked or ignored"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert private.read_bytes() == private_before


def test_update_adopts_an_identical_untracked_file_newly_tracked_upstream(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    media = source / "media" / "shared_12345678.svg"
    media.parent.mkdir()
    media.write_bytes(b"<svg>identical collaboration media</svg>\n")

    upstream = tmp_path / "identical-media-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Maintainer")
    upstream_media = upstream / "media" / media.name
    upstream_media.parent.mkdir()
    upstream_media.write_bytes(media.read_bytes())
    _commit(upstream, "Track exported collaboration media")
    _git(upstream, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    assert media.read_bytes() == upstream_media.read_bytes()
    assert _git(source, "ls-files", f"media/{media.name}").stdout.strip() == (
        f"media/{media.name}"
    )
    assert _git(source, "status", "--short").stdout == ""


@pytest.mark.parametrize("local_kind", ["different-bytes", "symlink", "mode-mismatch"])
def test_update_rejects_an_unsafe_untracked_file_match(
    tmp_path: Path, monkeypatch, local_kind
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    media = source / "media" / "shared_12345678.svg"
    media.parent.mkdir()
    upstream_bytes = b"<svg>upstream collaboration media</svg>\n"
    if local_kind == "different-bytes":
        media.write_bytes(b"<svg>different local media</svg>\n")
    elif local_kind == "symlink":
        target = media.with_name("local-target.svg")
        target.write_bytes(upstream_bytes)
        media.symlink_to(target.name)
    else:
        media.write_bytes(upstream_bytes)

    upstream = tmp_path / f"unsafe-media-upstream-{local_kind}"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Maintainer")
    upstream_media = upstream / "media" / media.name
    upstream_media.parent.mkdir()
    upstream_media.write_bytes(upstream_bytes)
    if local_kind == "mode-mismatch":
        upstream_media.chmod(0o755)
    _commit(upstream, "Track colliding collaboration media")
    _git(upstream, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="untracked or ignored"):
        run_update(SimpleNamespace(repository="owner/repo"))

    if local_kind == "symlink":
        assert media.is_symlink()
    elif local_kind == "different-bytes":
        assert media.read_bytes() == b"<svg>different local media</svg>\n"
    else:
        assert not media.stat().st_mode & 0o111


@pytest.mark.parametrize(
    ("local_path", "upstream_path"),
    [
        ("Media/Shared.svg", "media/shared.svg"),
        (
            "media/cafe\N{COMBINING ACUTE ACCENT}.svg",
            "media/caf\N{LATIN SMALL LETTER E WITH ACUTE}.svg",
        ),
    ],
    ids=["case-alias", "unicode-normalization-alias"],
)
def test_update_rejects_a_portable_alias_of_an_untracked_path(
    tmp_path: Path, monkeypatch, local_path, upstream_path
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = source / local_path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(b"private local bytes\n")

    upstream = tmp_path / f"portable-alias-{len(local_path)}"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Maintainer")
    candidate = upstream / upstream_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"public upstream bytes\n")
    _commit(upstream, "Track portable path alias")
    _git(upstream, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="untracked or ignored"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert private.read_bytes() == b"private local bytes\n"


def test_update_does_not_replace_an_untracked_directory_with_an_upstream_file(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = source / "local-data" / "private.txt"
    private.parent.mkdir()
    private.write_text("private nested bytes\n", encoding="utf-8")
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    private_before = private.read_bytes()

    attacker = tmp_path / "directory-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "local-data").write_text("upstream file\n", encoding="utf-8")
    _commit(attacker, "Replace local directory path upstream")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="local-data/private.txt.*local-data"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert private.read_bytes() == private_before


def test_update_does_not_replace_an_untracked_file_with_an_upstream_directory(
    tmp_path: Path, monkeypatch
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    private = source / "local-node"
    private.write_text("private file bytes\n", encoding="utf-8")
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    private_before = private.read_bytes()

    attacker = tmp_path / "file-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    (attacker / "local-node").mkdir()
    (attacker / "local-node" / "public.txt").write_text(
        "upstream nested file\n", encoding="utf-8"
    )
    _commit(attacker, "Replace local file path upstream")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="local-node.*local-node/public.txt"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    assert private.read_bytes() == private_before


@pytest.mark.parametrize("resolution_kind", ["symlink", "directory", "fifo"])
def test_conflict_retry_rejects_a_non_regular_editable_resolution(
    tmp_path: Path, monkeypatch, resolution_kind: str
):
    collection = _collection(tmp_path)
    remote = _safe_remote(tmp_path)
    source = _checkout_source(collection, remote)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("safe answer", "local answer"),
        encoding="utf-8",
    )
    attacker = tmp_path / "conflict-attacker"
    _git(tmp_path, "clone", str(remote), str(attacker))
    _configure(attacker, "Attacker")
    attacker_deck = attacker / "Deck.md"
    attacker_deck.write_text(
        attacker_deck.read_text(encoding="utf-8").replace(
            "safe answer", "upstream answer"
        ),
        encoding="utf-8",
    )
    _commit(attacker, "Edit deck upstream")
    _git(attacker, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="overlap"):
        run_update(SimpleNamespace(repository="owner/repo"))

    private = collection / "Private.md"
    private_before = private.read_bytes()
    deck_before = deck.read_bytes()
    head_before = _git(source, "rev-parse", "HEAD").stdout
    status_before = _git(source, "status", "--porcelain=v1", "-uall").stdout
    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.unlink()
    if resolution_kind == "symlink":
        conflict_file.symlink_to(private)
    elif resolution_kind == "directory":
        conflict_file.mkdir()
    else:
        os.mkfifo(conflict_file)

    with pytest.raises(ValueError, match="regular non-symlink file"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert conflict_file.is_symlink() or not conflict_file.is_file()
    assert private.read_bytes() == private_before
    assert deck.read_bytes() == deck_before
    assert _git(source, "rev-parse", "HEAD").stdout == head_before
    assert _git(source, "status", "--porcelain=v1", "-uall").stdout == status_before
    transactions = collection / ".ankiops" / "transactions"
    assert not transactions.exists() or not list(transactions.iterdir())
    state = SyncState.open(collection)
    try:
        operation = state.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "conflict"
        assert Path(str(operation["recovery_ref"])) == conflict_file.parent
    finally:
        state.close()

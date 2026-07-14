from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.collab.commands import run_publish, run_status, run_submit, run_update
from ankiops.collab.git_state import (
    CONFLICT_BASE_REF,
    CONFLICT_LOCAL_REF,
    CONFLICT_UPSTREAM_REF,
    CONTRIBUTION_BRANCH,
    INTEGRATED_REF,
    PUBLISH_DECK_CONFIG,
    PUBLISH_PREPARED_REF,
    SUBMISSION_REF,
    UPLOADED_REF,
)
from ankiops.collab.hosting import PullRequestInfo
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


def _configure(root: Path, name: str = "AnkiOps Test") -> None:
    _git(root, "config", "user.name", name)
    _git(root, "config", "user.email", f"{name.replace(' ', '.')}@example.test")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def _setup_collection(tmp_path: Path) -> Path:
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


def _setup_source(tmp_path: Path, collection: Path) -> tuple[Path, Path]:
    remote = tmp_path / "upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    (seed / "Deck.md").write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: upstream answer\n",
        encoding="utf-8",
    )
    _commit(seed, "Initial collab deck")
    _git(seed, "push", "origin", "main")

    source = collection / "collab" / "owner" / "repo"
    source.parent.mkdir(parents=True)
    _git(source.parent, "clone", "--origin", "upstream", str(remote), str(source))
    _configure(source, "Contributor")
    _git(source, "checkout", "-b", "ankiops/journal", "upstream/main")
    _git(source, "branch", "--unset-upstream")
    _git(source, "update-ref", INTEGRATED_REF, "upstream/main")
    return source, remote


@pytest.fixture(scope="session")
def collab_world_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("collab-world-template")
    collection = _setup_collection(root)
    _setup_source(root, collection)
    shutil.rmtree(root / "seed")
    return root


@pytest.fixture
def collab_world(tmp_path, monkeypatch, collab_world_template):
    shutil.copytree(collab_world_template, tmp_path, dirs_exist_ok=True)
    collection = tmp_path / "collection"
    source = collection / "collab" / "owner" / "repo"
    remote = tmp_path / "upstream.git"
    _git(source, "remote", "set-url", "upstream", str(remote))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    return collection, source, remote


def _upstream_edit(
    tmp_path: Path,
    remote: Path,
    old: str,
    new: str,
    *,
    filename: str = "Deck.md",
) -> str:
    clone = tmp_path / f"upstream-{len(list(tmp_path.iterdir()))}"
    _git(tmp_path, "clone", str(remote), str(clone))
    _configure(clone, "Other Contributor")
    path = clone / filename
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    head = _commit(clone, f"Change {old} to {new}")
    _git(clone, "push", "origin", "main")
    return head


def _conflict_root(collection: Path) -> Path:
    return collection / ".ankiops" / "conflicts" / "owner" / "repo"


def test_clean_update_is_a_noop(collab_world, capsys):
    _collection, source, _remote = collab_world
    before = _git(source, "rev-parse", "HEAD").stdout.strip()

    run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert "already up to date" in capsys.readouterr().out


def test_update_commits_staged_unstaged_and_untracked_changes(collab_world, tmp_path):
    _collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    _git(source, "add", "Deck.md")
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("local answer", "final answer"),
        encoding="utf-8",
    )
    (source / "notes.txt").write_text("untracked contribution\n", encoding="utf-8")
    upstream_clone = tmp_path / "nonconflicting-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream_clone))
    _configure(upstream_clone, "Other Contributor")
    (upstream_clone / "README.md").write_text("upstream docs\n", encoding="utf-8")
    upstream = _commit(upstream_clone, "Add upstream documentation")
    _git(upstream_clone, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "status", "--short").stdout == ""
    assert _git(source, "ls-files", "notes.txt").stdout.strip() == "notes.txt"
    assert "final answer" in deck.read_text(encoding="utf-8")
    assert (source / "README.md").read_text() == "upstream docs\n"
    assert _git(source, "rev-parse", INTEGRATED_REF).stdout.strip() == upstream


def test_text_conflict_commits_local_checkpoint_and_preserves_evidence(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    before = _git(source, "rev-parse", "HEAD").stdout.strip()
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    (source / "draft.txt").write_text("keep me\n", encoding="utf-8")
    upstream = _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")

    with pytest.raises(ValueError, match="Local contribution committed"):
        run_update(SimpleNamespace(repository="owner/repo"))

    local = _git(source, "rev-parse", "HEAD").stdout.strip()
    root = _conflict_root(collection)
    assert local != before
    assert _git(source, "status", "--short").stdout == ""
    assert _git(source, "ls-files", "draft.txt").stdout.strip() == "draft.txt"
    assert json.loads((root / "conflict.json").read_text()) == {
        "kind": "update",
        "requested_title": None,
    }
    assert (root / "Deck.md.base").exists()
    assert (root / "Deck.md.local").exists()
    assert (root / "Deck.md.upstream").exists()
    assert _git(source, "rev-parse", CONFLICT_LOCAL_REF).stdout.strip() == local
    assert _git(source, "rev-parse", CONFLICT_UPSTREAM_REF).stdout.strip() == upstream
    assert _git(source, "rev-parse", CONFLICT_BASE_REF).stdout.strip()


def test_conflict_retry_uses_frozen_upstream_then_recommends_next_update(
    collab_world, tmp_path, capsys
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    frozen = _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    later = tmp_path / "later"
    _git(tmp_path, "clone", str(remote), str(later))
    _configure(later, "Later Contributor")
    (later / "README.md").write_text("later update\n", encoding="utf-8")
    newest = _commit(later, "Later upstream update")
    _git(later, "push", "origin", "main")
    (_conflict_root(collection) / "Deck.md").write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: resolved answer\n",
        encoding="utf-8",
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert _git(source, "rev-parse", INTEGRATED_REF).stdout.strip() == frozen
    assert "resolved frozen conflict" in output
    assert "Integrate the newer update: ankiops collab update owner/repo" in output
    assert not _conflict_root(collection).exists()

    run_update(SimpleNamespace(repository="owner/repo"))
    assert _git(source, "rev-parse", INTEGRATED_REF).stdout.strip() == newest


def test_submit_conflict_keeps_requested_title_across_newer_update(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, remote = collab_world
    publish = tmp_path / "title-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    frozen = _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")

    with pytest.raises(ValueError, match="Local contribution committed"):
        run_submit(
            SimpleNamespace(repository="owner/repo", title="Keep this exact title")
        )

    later = tmp_path / "title-later"
    _git(tmp_path, "clone", str(remote), str(later))
    _configure(later)
    (later / "README.md").write_text("newer update\n", encoding="utf-8")
    newest = _commit(later, "Newer upstream update")
    _git(later, "push", "origin", "main")
    (_conflict_root(collection) / "Deck.md").write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: resolved answer\n",
        encoding="utf-8",
    )

    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    repository = GitRepository(source)
    snapshot = repository.ref_sha(SUBMISSION_REF)
    assert snapshot is not None
    assert repository.commit_message(snapshot).splitlines()[0] == (
        "Keep this exact title"
    )
    assert repository.ref_sha(INTEGRATED_REF) == frozen
    assert repository.remote_url("publish") is None
    output = capsys.readouterr().out
    assert "submission prepared locally" in output
    assert "Integrate the newer update: ankiops collab update owner/repo" in output

    run_update(SimpleNamespace(repository="owner/repo"))
    assert repository.ref_sha(INTEGRATED_REF) == newest
    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert pull["state"] == "OPEN"
    assert pull["title"] == "Keep this exact title"


def test_source_edits_are_rejected_while_conflict_is_pending(collab_world, tmp_path):
    _collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))
    deck.write_text(
        deck.read_text(encoding="utf-8") + "source edit\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Do not edit the subscribed source"):
        run_update(SimpleNamespace(repository="owner/repo"))


def test_binary_conflict_uses_placeholder_and_raw_evidence(collab_world, tmp_path):
    collection, source, remote = collab_world
    seed = tmp_path / "binary-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed)
    (seed / "asset.bin").write_bytes(b"base\x00bytes")
    _commit(seed, "Add binary")
    _git(seed, "push", "origin", "main")
    run_update(SimpleNamespace(repository="owner/repo"))
    (source / "asset.bin").write_bytes(b"local\x00bytes")
    upstream = tmp_path / "binary-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream)
    (upstream / "asset.bin").write_bytes(b"remote\x00bytes")
    _commit(upstream, "Change binary")
    _git(upstream, "push", "origin", "main")

    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    root = _conflict_root(collection)
    assert b"AnkiOps unresolved conflict" in (root / "asset.bin").read_bytes()
    assert (root / "asset.bin.local").read_bytes() == b"local\x00bytes"
    assert (root / "asset.bin.upstream").read_bytes() == b"remote\x00bytes"


def test_delete_modify_conflict_can_resolve_to_deletion(collab_world, tmp_path):
    collection, source, remote = collab_world
    seed = tmp_path / "delete-seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed)
    (seed / "notes.txt").write_text("base\n", encoding="utf-8")
    _commit(seed, "Add auxiliary notes")
    _git(seed, "push", "origin", "main")
    run_update(SimpleNamespace(repository="owner/repo"))
    (source / "notes.txt").unlink()
    _upstream_edit(tmp_path, remote, "base", "remote edit", filename="notes.txt")

    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))
    resolution = _conflict_root(collection) / "notes.txt"
    resolution.unlink()
    run_update(SimpleNamespace(repository="owner/repo"))

    assert not (source / "notes.txt").exists()
    assert _git(source, "status", "--short").stdout == ""


def test_rename_conflict_preserves_both_names_and_accepts_one(collab_world, tmp_path):
    collection, source, remote = collab_world
    _git(source, "mv", "Deck.md", "Local.md")
    upstream = tmp_path / "rename-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream)
    _git(upstream, "mv", "Deck.md", "Remote.md")
    _commit(upstream, "Rename deck remotely")
    _git(upstream, "push", "origin", "main")

    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    root = _conflict_root(collection)
    assert list(root.glob("Local.md.*"))
    assert list(root.glob("Remote.md.*"))
    local_resolution = root / "Local.md"
    local_resolution.write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: kept local name\n",
        encoding="utf-8",
    )
    (root / "Remote.md").unlink(missing_ok=True)
    (root / "Deck.md").unlink(missing_ok=True)
    run_update(SimpleNamespace(repository="owner/repo"))

    assert (source / "Local.md").exists()
    assert not (source / "Remote.md").exists()
    assert not (source / "Deck.md").exists()


def _setup_publish_deck(tmp_path: Path) -> tuple[Path, Path, Path]:
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: publish-key -->\nQ: publish\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add publish deck")
    state = SyncState.open(collection)
    state.upsert_note_links([("publish-key", 123)])
    state.upsert_deck("Deck", 456)
    state.close()
    remote = tmp_path / "published.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    return collection, deck, remote


def _redirect_publish_remote(monkeypatch, remote: Path) -> None:
    original = GitRepository.set_remote

    def local_remote(repository, name, _url):
        original(repository, name, str(remote))

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)


def test_publish_handoff_is_git_native_and_clears_preparation(
    tmp_path, monkeypatch, capsys
):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    source = GitRepository(collection / "collab" / "owner" / "repo")
    assert not deck.exists()
    assert source.ref_sha(PUBLISH_PREPARED_REF) is None
    assert source.config_get(PUBLISH_DECK_CONFIG) is None
    assert source.ref_sha(INTEGRATED_REF) == source.head()
    assert _git(remote, "rev-parse", "main").stdout.strip() == source.head()
    assert "ankiops fa" not in capsys.readouterr().out


def test_publish_retry_adopts_completed_push(tmp_path, monkeypatch):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    original_push = GitRepository.push
    failed = False

    def push_then_interrupt(repository, remote_name, source_ref, branch):
        nonlocal failed
        original_push(repository, remote_name, source_ref, branch)
        if not failed:
            failed = True
            raise subprocess.CalledProcessError(1, ["git", "push"])

    monkeypatch.setattr(GitRepository, "push", push_then_interrupt)
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Retry exactly: ankiops collab publish"):
        run_publish(args)
    source = GitRepository(collection / "collab" / "owner" / "repo")
    prepared = source.ref_sha(PUBLISH_PREPARED_REF)
    assert prepared == _git(remote, "rev-parse", "main").stdout.strip()

    run_publish(args)
    assert not deck.exists()
    assert source.ref_sha(PUBLISH_PREPARED_REF) is None


def test_status_reports_prepared_publish_and_exact_retry(tmp_path, monkeypatch, capsys):
    collection, _deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    monkeypatch.setattr(
        GitRepository,
        "push",
        lambda *_args: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "push"])
        ),
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")
    with pytest.raises(ValueError):
        run_publish(args)

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Publish: prepared; handoff incomplete" in output
    assert "Retry publish: ankiops collab publish Deck owner/repo" in output


def test_publish_retry_finishes_interrupted_ownership_transfer(tmp_path, monkeypatch):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    from ankiops.collab import publish as publish_module

    original_transfer = publish_module._transfer_sync_ownership
    failed = False

    def fail_once(root, plan):
        nonlocal failed
        if not failed:
            failed = True
            raise ValueError("simulated ownership interruption")
        original_transfer(root, plan)

    monkeypatch.setattr(publish_module, "_transfer_sync_ownership", fail_once)
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Prepared repository"):
        run_publish(args)
    assert not deck.exists()
    run_publish(args)
    state = SyncState.open(collection)
    try:
        assert state.resolve_note_sources(["publish-key"])["publish-key"] == (
            "collab/owner/repo"
        )
    finally:
        state.close()


def test_publish_retry_finishes_after_root_removal(tmp_path, monkeypatch):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    original_commit_paths = GitRepository.commit_paths
    interrupted = False

    def interrupt_collection_commit(repository, paths, message):
        nonlocal interrupted
        if repository.root == collection and not interrupted:
            interrupted = True
            raise ValueError("simulated interruption after root removal")
        return original_commit_paths(repository, paths, message)

    monkeypatch.setattr(GitRepository, "commit_paths", interrupt_collection_commit)
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Prepared repository"):
        run_publish(args)
    assert not deck.exists()

    run_publish(args)
    source = GitRepository(collection / "collab" / "owner" / "repo")
    assert source.ref_sha(PUBLISH_PREPARED_REF) is None


def test_publish_retry_finishes_interrupted_preparation_cleanup(tmp_path, monkeypatch):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    _redirect_publish_remote(monkeypatch, remote)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    original_unset = GitRepository.config_unset
    interrupted = False

    def interrupt_cleanup(repository, key):
        nonlocal interrupted
        if key == PUBLISH_DECK_CONFIG and not interrupted:
            interrupted = True
            raise ValueError("simulated interruption during preparation cleanup")
        return original_unset(repository, key)

    monkeypatch.setattr(GitRepository, "config_unset", interrupt_cleanup)
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Prepared repository"):
        run_publish(args)
    source = GitRepository(collection / "collab" / "owner" / "repo")
    assert source.ref_sha(PUBLISH_PREPARED_REF) is None
    assert source.config_get(PUBLISH_DECK_CONFIG) == "Deck"
    assert not deck.exists()

    run_publish(args)
    assert source.config_get(PUBLISH_DECK_CONFIG) is None


def test_publish_never_overwrites_unrelated_local_or_remote_repository(
    tmp_path, monkeypatch
):
    collection, deck, remote = _setup_publish_deck(tmp_path)
    source = collection / "collab" / "owner" / "repo"
    source.mkdir(parents=True)
    (source / "mine.txt").write_text("mine\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    with pytest.raises(ValueError, match="not a Git repository"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))
    assert deck.exists()
    assert (source / "mine.txt").exists()

    shutil.rmtree(source)
    unrelated = tmp_path / "unrelated"
    _git(tmp_path, "clone", str(remote), str(unrelated))
    _configure(unrelated)
    (unrelated / "unrelated.txt").write_text("remote data\n", encoding="utf-8")
    _commit(unrelated, "Unrelated history")
    _git(unrelated, "push", "origin", "main")
    _redirect_publish_remote(monkeypatch, remote)
    with pytest.raises(ValueError, match="unrelated content"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))
    assert deck.exists()
    assert (source / "Deck.md").exists()


def _mock_submission_github(monkeypatch, publish: Path):
    original_set_remote = GitRepository.set_remote
    pull = {
        "state": None,
        "url": "https://github.test/owner/repo/pull/1",
        "title": None,
    }

    def local_publish(repository, name, _url):
        original_set_remote(repository, name, str(publish))

    def remote_sha() -> str:
        result = _git(
            publish, "rev-parse", f"refs/heads/{CONTRIBUTION_BRANCH}", check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    def pull_request(*_args):
        if pull["state"] is None:
            return None
        return PullRequestInfo(
            url=str(pull["url"]),
            state=str(pull["state"]),
            head_branch=CONTRIBUTION_BRANCH,
            head_sha=remote_sha(),
            head_repository="contributor/repo",
        )

    def create_pr(*_args, **kwargs):
        pull["state"] = "OPEN"
        pull["title"] = kwargs["title"]
        return pull["url"]

    def update_pr(_self, _url, *, title):
        pull["title"] = title

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands._github_slug_from_remote",
        lambda url: "contributor/repo" if url == str(publish) else None,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.find_pull_request", pull_request
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.update_pr", update_pr)
    return pull


def test_submit_commits_all_nonignored_changes_to_deterministic_branch(
    collab_world, tmp_path, monkeypatch, capsys
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    _git(source, "add", "Deck.md")
    (source / "draft.txt").write_text("untracked\n", encoding="utf-8")
    (source / ".gitignore").write_text("scratch.tmp\n", encoding="utf-8")
    (source / "scratch.tmp").write_text("ignored\n", encoding="utf-8")
    args = SimpleNamespace(repository="owner/repo", title="Improve shared deck")

    run_submit(args)

    repository = GitRepository(source)
    remote = repository.remote_branch_sha("publish", CONTRIBUTION_BRANCH)
    assert repository.ref_sha(SUBMISSION_REF) == remote
    assert repository.ref_sha(UPLOADED_REF) == remote
    assert repository.commit_message(SUBMISSION_REF).splitlines()[0] == (
        "Improve shared deck"
    )
    assert _git(publish, "show", f"{remote}:draft.txt").stdout == "untracked\n"
    assert _git(publish, "show", f"{remote}:scratch.tmp", check=False).returncode != 0
    assert pull["state"] == "OPEN"
    assert "pull request opened" in capsys.readouterr().out

    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert "pull request already open" in capsys.readouterr().out


def test_submit_retry_adopts_push_completed_before_interruption(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    original_push = GitRepository.push
    interrupted = False

    def push_then_interrupt(repository, remote_name, source_ref, branch):
        nonlocal interrupted
        original_push(repository, remote_name, source_ref, branch)
        if branch == CONTRIBUTION_BRANCH and not interrupted:
            interrupted = True
            raise subprocess.CalledProcessError(1, ["git", "push"])

    monkeypatch.setattr(GitRepository, "push", push_then_interrupt)
    args = SimpleNamespace(repository="owner/repo", title="Retry title")

    with pytest.raises(ValueError, match="Submission prepared locally"):
        run_submit(args)
    repository = GitRepository(source)
    assert repository.ref_sha(UPLOADED_REF) is None
    assert repository.remote_branch_sha("publish", CONTRIBUTION_BRANCH) == (
        repository.ref_sha(SUBMISSION_REF)
    )

    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert repository.ref_sha(UPLOADED_REF) == repository.ref_sha(SUBMISSION_REF)
    assert pull["state"] == "OPEN"


def test_submit_retry_reuses_pr_created_before_interruption(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    interrupted = False

    def create_then_interrupt(*_args, **kwargs):
        nonlocal interrupted
        pull["state"] = "OPEN"
        pull["title"] = kwargs["title"]
        if not interrupted:
            interrupted = True
            raise ValueError("lost response after creation")
        return pull["url"]

    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr", create_then_interrupt
    )
    args = SimpleNamespace(repository="owner/repo", title="Recovered PR")

    with pytest.raises(ValueError, match="Submission pushed"):
        run_submit(args)
    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert pull["state"] == "OPEN"


def test_new_submission_retry_ignores_historic_merged_pr(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, upstream_remote = collab_world
    publish = tmp_path / "reused-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "first answer"),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="First round"))
    first_snapshot = GitRepository(source).ref_sha(SUBMISSION_REF)

    merged = tmp_path / "first-merged"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged)
    merged_deck = merged / "Deck.md"
    merged_deck.write_text(
        merged_deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first answer"
        ),
        encoding="utf-8",
    )
    _commit(merged, "Merge first contribution")
    _git(merged, "push", "origin", "main")
    pull["state"] = "MERGED"
    run_update(SimpleNamespace(repository="owner/repo"))

    deck.write_text(
        deck.read_text(encoding="utf-8").replace("first answer", "second answer"),
        encoding="utf-8",
    )
    original_push = GitRepository.push
    interrupted = False

    def interrupt_before_push(repository, remote_name, source_ref, branch):
        nonlocal interrupted
        if branch == CONTRIBUTION_BRANCH and not interrupted:
            interrupted = True
            raise subprocess.CalledProcessError(1, ["git", "push"])
        return original_push(repository, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "push", interrupt_before_push)
    with pytest.raises(ValueError, match="Submission prepared locally"):
        run_submit(SimpleNamespace(repository="owner/repo", title="Second round"))

    repository = GitRepository(source)
    second_snapshot = repository.ref_sha(SUBMISSION_REF)
    assert second_snapshot != first_snapshot
    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert pull["state"] == "OPEN"
    assert pull["title"] == "Second round"


def test_closed_submission_branch_is_deleted_with_lease_before_reuse(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "closed-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Closed round"))
    repository = GitRepository(source)
    uploaded = repository.ref_sha(UPLOADED_REF)
    pull["state"] = "CLOSED"
    deleted_with = []
    original_delete = GitRepository.delete_remote_branch_with_lease

    def record_delete(repository, remote_name, branch, expected_sha):
        deleted_with.append(expected_sha)
        return original_delete(repository, remote_name, branch, expected_sha)

    monkeypatch.setattr(
        GitRepository,
        "delete_remote_branch_with_lease",
        record_delete,
    )

    run_submit(SimpleNamespace(repository="owner/repo", title="Replacement round"))

    assert deleted_with == [uploaded]
    assert pull["state"] == "OPEN"
    assert pull["title"] == "Replacement round"
    assert repository.ref_sha(UPLOADED_REF) == repository.ref_sha(SUBMISSION_REF)


def test_submit_never_overwrites_remotely_changed_branch(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    attacker = tmp_path / "branch-editor"
    _git(tmp_path, "clone", str(publish), str(attacker))
    _configure(attacker)
    _git(attacker, "checkout", CONTRIBUTION_BRANCH)
    (attacker / "README.md").write_text("changed remotely\n", encoding="utf-8")
    changed = _commit(attacker, "Remote branch edit")
    _git(attacker, "push", "origin", CONTRIBUTION_BRANCH)
    deck.write_text(deck.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed remotely"):
        run_submit(SimpleNamespace(repository="owner/repo", title=None))
    assert _git(publish, "rev-parse", CONTRIBUTION_BRANCH).stdout.strip() == changed


def test_status_reports_open_revised_and_remotely_changed_submission(
    collab_world, tmp_path, monkeypatch, capsys
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    capsys.readouterr()

    run_status(SimpleNamespace(repository="owner/repo"))
    assert "Submission: pull request open" in capsys.readouterr().out

    deck.write_text(deck.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    run_status(SimpleNamespace(repository="owner/repo"))
    assert "Update pull request: ankiops collab submit owner/repo" in (
        capsys.readouterr().out
    )

    _git(source, "restore", "Deck.md")
    attacker = tmp_path / "status-branch-editor"
    _git(tmp_path, "clone", str(publish), str(attacker))
    _configure(attacker)
    _git(attacker, "checkout", CONTRIBUTION_BRANCH)
    (attacker / "README.md").write_text("changed remotely\n", encoding="utf-8")
    _commit(attacker, "Remote branch edit")
    _git(attacker, "push", "origin", CONTRIBUTION_BRANCH)
    run_status(SimpleNamespace(repository="owner/repo"))
    output = capsys.readouterr().out
    assert "Submission: changed remotely; protected" in output
    assert "Inspect the remote contribution" in output


def test_merged_submission_updates_from_uploaded_base_and_cleans_refs(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, upstream_remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    pull = _mock_submission_github(monkeypatch, publish)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title=None))
    merged = tmp_path / "merged"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged)
    merged_deck = merged / "Deck.md"
    merged_deck.write_text(
        merged_deck.read_text(encoding="utf-8").replace(
            "upstream answer", "submitted answer"
        ),
        encoding="utf-8",
    )
    _commit(merged, "Merge contribution")
    _git(merged, "push", "origin", "main")
    pull["state"] = "MERGED"

    run_update(SimpleNamespace(repository="owner/repo"))

    repository = GitRepository(source)
    assert repository.ref_sha(SUBMISSION_REF) is None
    assert repository.ref_sha(UPLOADED_REF) is None
    assert repository.remote_branch_sha("publish", CONTRIBUTION_BRANCH) is None
    assert "submitted answer" in deck.read_text(encoding="utf-8")


def test_status_reports_frozen_conflict_and_exact_retry(collab_world, tmp_path, capsys):
    _collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Conflict: frozen update" in output
    assert "Resolve and retry: ankiops collab update owner/repo" in output

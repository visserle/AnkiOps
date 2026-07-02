from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.git import RepositoryGit
from ankiops.shared.commands import (
    run_publish,
    run_status,
    run_submit,
    run_subscribe,
    run_update,
)
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
        "/shared/\n.ankiops.db\n.ankiops.db-shm\n.ankiops.db-wal\n.ankiops/\n",
        encoding="utf-8",
    )
    (collection / "Private.md").write_text("private baseline\n", encoding="utf-8")
    _commit(collection, "Initialize private collection")
    db = SyncState.open(collection)
    db.set_profile_name("Test")
    db.close()
    return collection


def _setup_source(tmp_path: Path, collection: Path) -> tuple[Path, Path]:
    remote = tmp_path / "upstream.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _configure(seed, "Upstream User")
    DeckFileHarness().eject_default_note_types(seed / "note_types")
    (seed / "Deck.md").write_text(
        "<!-- note_key: shared-key -->\nQ: shared question\nA: upstream answer\n",
        encoding="utf-8",
    )
    _commit(seed, "Initial shared deck")
    _git(seed, "push", "origin", "main")

    source = collection / "shared" / "owner" / "repo"
    source.parent.mkdir(parents=True)
    _git(source.parent, "clone", "--origin", "upstream", str(remote), str(source))
    _configure(source, "Contributor")
    _git(source, "checkout", "-b", "ankiops/work", "upstream/main")
    return source, remote


@pytest.fixture
def shared_world(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    source, remote = _setup_source(tmp_path, collection)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir", lambda: collection
    )
    return collection, source, remote


def _upstream_edit(tmp_path: Path, remote: Path, old: str, new: str) -> str:
    clone = tmp_path / f"upstream-edit-{new.replace(' ', '-')}"
    _git(tmp_path, "clone", str(remote), str(clone))
    _configure(clone, "Other Contributor")
    deck = clone / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    head = _commit(clone, f"Change {old} to {new}")
    _git(clone, "push", "origin", "main")
    return head


def test_update_noop_creates_no_commit(shared_world):
    _collection, source, _remote = shared_world
    before = _git(source, "rev-parse", "HEAD").stdout.strip()

    run_update(SimpleNamespace(repo="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(source, "branch", "--list", "ankiops/recovery/*").stdout == ""


def test_submit_noop_creates_no_commit_branch_push_or_pr(shared_world, monkeypatch):
    collection, source, _remote = shared_world
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_branches = _git(source, "branch", "--format=%(refname)").stdout
    publish = monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: pytest.fail("no-op submit must not publish"),
    )
    assert publish is None

    run_submit(SimpleNamespace(repo="owner/repo", message=None))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "branch", "--format=%(refname)").stdout == before_branches
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("owner/repo") is None
    finally:
        db.close()


def test_subscribe_clones_independent_repository(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    _source, remote = _setup_source(tmp_path, collection)
    shutil_source = collection / "shared" / "owner" / "repo"
    shutil.rmtree(shutil_source)
    original_clone = RepositoryGit.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream"):
        return original_clone(cls, str(remote_path), target, remote=remote)

    remote_path = remote
    monkeypatch.setattr(RepositoryGit, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir", lambda: collection
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.ensure_authenticated", lambda *_: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.repo_info",
        lambda *_: {"default_branch": "main"},
    )

    run_subscribe(SimpleNamespace(repo="owner/repo"))

    assert RepositoryGit(shutil_source).is_repo()
    assert _git(shutil_source, "branch", "--show-current").stdout.strip() == (
        "ankiops/work"
    )
    assert _git(collection, "status", "--short").stdout == ""


def test_publish_moves_deck_into_independent_repository(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: create-key -->\nQ: create\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add private deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("create-key", 123)])
    db.upsert_deck("Deck", 456)
    db.close()
    remote = tmp_path / "created.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = RepositoryGit.set_remote

    def local_remote(repo, name, _url):
        original_set_remote(repo, name, str(remote))

    monkeypatch.setattr(RepositoryGit, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.shared.create.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir", lambda: collection
    )

    run_publish(SimpleNamespace(deck="Deck", repo="owner/repo", public=False))

    source = collection / "shared" / "owner" / "repo"
    assert not deck.exists()
    assert RepositoryGit(source).is_repo()
    assert "shared/owner/repo/AnkiOpsQA" in (source / "Deck.md").read_text(
        encoding="utf-8"
    )
    assert _git(source, "branch", "--show-current").stdout.strip() == "ankiops/work"
    assert _git(remote, "rev-parse", "refs/heads/main").stdout.strip()
    db = SyncState.open(collection)
    try:
        assert db.resolve_note_sources(["create-key"])["create-key"] == "owner/repo"
        assert db.resolve_deck_source(456) == "owner/repo"
    finally:
        db.close()


def test_publish_retry_reuses_local_repository_after_push_failure(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: retry-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add retry deck")
    remote = tmp_path / "retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = RepositoryGit.set_remote
    original_push = RepositoryGit.push
    attempts = 0

    def local_remote(repo, name, _url):
        original_set_remote(repo, name, str(remote))

    def fail_once(repo, remote_name, source_ref, branch):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(1, ["git", "push"])
        original_push(repo, remote_name, source_ref, branch)

    monkeypatch.setattr(RepositoryGit, "set_remote", local_remote)
    monkeypatch.setattr(RepositoryGit, "push", fail_once)
    monkeypatch.setattr(
        "ankiops.shared.create.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repo="owner/repo", public=False)

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_publish(args)
    source = collection / "shared" / "owner" / "repo"
    first_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    run_publish(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == first_head
    assert _git(source, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert not deck.exists()


def test_update_checkpoints_local_edit_and_integrates_remote(shared_world, tmp_path):
    collection, source, remote = shared_world
    private_before = (collection / "Private.md").read_bytes()
    (source / "Local.md").write_text(
        "<!-- note_key: local-key -->\nQ: local\nA: edit\n", encoding="utf-8"
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")

    run_update(SimpleNamespace(repo="owner/repo"))

    assert (source / "Local.md").exists()
    assert "remote answer" in (source / "Deck.md").read_text(encoding="utf-8")
    assert (collection / "Private.md").read_bytes() == private_before
    assert (
        "Save local deck changes for owner/repo"
        in _git(source, "log", "--format=%s").stdout
    )


def test_conflict_preserves_versions_and_leaves_source_unchanged(
    shared_world, tmp_path
):
    collection, source, remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    _git(source, "add", "Deck.md")
    (source / "Draft.md").write_text("unstaged draft\n", encoding="utf-8")
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_status = _git(source, "status", "--porcelain=v1").stdout
    before_deck = deck.read_bytes()
    _upstream_edit(tmp_path, remote, "upstream answer", "github answer")

    with pytest.raises(ValueError, match="failure"):
        run_update(SimpleNamespace(repo="owner/repo"))

    assert deck.read_bytes() == before_deck
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    conflicts = collection / ".ankiops" / "conflicts"
    assert list(conflicts.rglob("Deck.md"))
    assert list(conflicts.rglob("Deck.md.local"))
    assert list(conflicts.rglob("Deck.md.upstream"))
    assert list(conflicts.rglob("Deck.md.base"))


def test_conflict_can_be_resolved_by_editing_preserved_markdown_and_retrying_update(
    shared_world, tmp_path
):
    collection, source, remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "github answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repo="owner/repo"))
    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: shared-key -->\nQ: shared question\nA: combined answer\n",
        encoding="utf-8",
    )

    run_update(SimpleNamespace(repo="owner/repo"))

    assert not RepositoryGit(source).unmerged_paths()
    assert "combined answer" in deck.read_text(encoding="utf-8")
    assert not list((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("owner/repo") is None
    finally:
        db.close()


def test_submit_conflict_is_resumed_with_update(shared_world, tmp_path):
    collection, source, remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "local contribution"
        ),
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    with pytest.raises(ValueError, match="shared update owner/repo"):
        run_submit(SimpleNamespace(repo="owner/repo", message=None))

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: shared-key -->\nQ: shared question\nA: combined\n",
        encoding="utf-8",
    )
    run_update(SimpleNamespace(repo="owner/repo"))

    assert "combined" in deck.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("owner/repo") is None
    finally:
        db.close()


def test_failed_update_fetch_leaves_repository_exactly_unchanged(
    shared_world, monkeypatch
):
    collection, source, _remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local edit"),
        encoding="utf-8",
    )
    _git(source, "add", "Deck.md")
    before_head = _git(source, "rev-parse", "HEAD").stdout
    before_status = _git(source, "status", "--porcelain=v1").stdout
    before_bytes = deck.read_bytes()
    monkeypatch.setattr(
        RepositoryGit,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "fetch"])
        ),
    )

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_update(SimpleNamespace(repo="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    assert deck.read_bytes() == before_bytes
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("owner/repo")
        assert operation is not None
        assert operation["state"] == "failed"
    finally:
        db.close()


def test_submit_commits_only_source_and_reuses_operation(
    shared_world, tmp_path, monkeypatch
):
    collection, source, _remote = shared_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    private = collection / "Private.md"
    private.write_text("private dirty\n", encoding="utf-8")
    (source / "Deck.md").write_text(
        (source / "Deck.md")
        .read_text(encoding="utf-8")
        .replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )

    original_set_remote = RepositoryGit.set_remote

    def local_publish(repo, name, _url):
        original_set_remote(repo, name, str(publish))

    monkeypatch.setattr(RepositoryGit, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    created_pr = []

    def create_pr(*_args, **kwargs):
        created_pr.append(kwargs)
        return "https://github.test/owner/repo/pull/1"

    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        create_pr,
    )

    args = SimpleNamespace(repo="owner/repo", message="Clarify shared answer")
    run_submit(args)
    first_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    first_refs = _git(publish, "for-each-ref", "--format=%(refname)").stdout
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == first_head
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == first_refs
    assert created_pr[0]["head"].startswith("contributor:ankiops/")
    assert private.read_text(encoding="utf-8") == "private dirty\n"
    assert _git(collection, "status", "--short").stdout.strip() == "M Private.md"
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/1"
    finally:
        db.close()


def test_submit_retry_after_failed_pr_reuses_pushed_branch(
    shared_world, tmp_path, monkeypatch
):
    collection, source, _remote = shared_world
    publish = tmp_path / "pr-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "PR retry answer"),
        encoding="utf-8",
    )
    original_set_remote = RepositoryGit.set_remote

    def local_publish(repo, name, _url):
        original_set_remote(repo, name, str(publish))

    calls = 0

    def fail_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("simulated PR failure")
        return "https://github.test/owner/repo/pull/2"

    monkeypatch.setattr(RepositoryGit, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.shared.commands.GitHubHost.create_pr", fail_once)
    args = SimpleNamespace(repo="owner/repo", message="Retry PR")

    with pytest.raises(ValueError, match="reached GitHub"):
        run_submit(args)
    refs_after_failure = _git(
        publish, "for-each-ref", "--format=%(refname):%(objectname)"
    ).stdout
    head_after_failure = _git(source, "rev-parse", "HEAD").stdout.strip()
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == head_after_failure
    assert (
        _git(publish, "for-each-ref", "--format=%(refname):%(objectname)").stdout
        == refs_after_failure
    )
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/2"
    finally:
        db.close()


def test_submit_retry_after_failed_push_reuses_commit_and_branch(
    shared_world, tmp_path, monkeypatch
):
    collection, source, _remote = shared_world
    publish = tmp_path / "push-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "push retry answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = RepositoryGit.set_remote
    original_push = RepositoryGit.push
    pushes = 0

    def local_publish(repo, name, _url):
        original_set_remote(repo, name, str(publish))

    def fail_once(repo, remote_name, source_ref, branch):
        nonlocal pushes
        pushes += 1
        if pushes == 1:
            raise subprocess.CalledProcessError(1, ["git", "push"])
        original_push(repo, remote_name, source_ref, branch)

    monkeypatch.setattr(RepositoryGit, "set_remote", local_publish)
    monkeypatch.setattr(RepositoryGit, "push", fail_once)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/3",
    )
    args = SimpleNamespace(repo="owner/repo", message="Retry push")

    with pytest.raises(ValueError, match="did not reach GitHub"):
        run_submit(args)
    failed_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    db = SyncState.open(collection)
    try:
        failed_operation = db.get_shared_operation("owner/repo")
        assert failed_operation is not None
        branch = failed_operation["publish_branch"]
        assert failed_operation["state"] == "push_failed"
    finally:
        db.close()
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == failed_head
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("owner/repo")
        assert operation is not None
        assert operation["publish_branch"] == branch
        assert operation["state"] == "pr_open"
    finally:
        db.close()


def test_update_after_squash_merge_cleans_submission_state(
    shared_world, tmp_path, monkeypatch
):
    collection, source, upstream_remote = shared_world
    publish = tmp_path / "merged-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "merged answer"),
        encoding="utf-8",
    )
    original_set_remote = RepositoryGit.set_remote

    def local_publish(repo, name, _url):
        original_set_remote(repo, name, str(publish))

    monkeypatch.setattr(RepositoryGit, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/4",
    )
    run_submit(SimpleNamespace(repo="owner/repo", message="Merged change"))

    merged = tmp_path / "squash-merge"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged, "Maintainer")
    (merged / "Deck.md").write_bytes(deck.read_bytes())
    _commit(merged, "Squash merged contribution")
    _git(merged, "push", "origin", "main")

    run_update(SimpleNamespace(repo="owner/repo"))

    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("owner/repo") is None
    finally:
        db.close()
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == ""


def test_status_reports_local_state_when_fetch_fails(shared_world, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    (source / "Draft.md").write_text("draft\n", encoding="utf-8")
    monkeypatch.setattr(
        RepositoryGit,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "fetch"])
        ),
    )

    run_status(SimpleNamespace(repo="owner/repo"))

    assert "Local shared changes: 1" in caplog.text
    assert "local files were not changed" in caplog.text
    assert "Private deck changes: 0" in caplog.text

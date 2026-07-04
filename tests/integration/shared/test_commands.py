from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.git import GitRepository
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


@pytest.fixture(scope="session")
def shared_world_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("shared-world-template")
    collection = _setup_collection(root)
    _setup_source(root, collection)
    shutil.rmtree(root / "seed")
    return root


@pytest.fixture
def shared_world(tmp_path, monkeypatch, shared_world_template):
    shutil.copytree(shared_world_template, tmp_path, dirs_exist_ok=True)
    collection = tmp_path / "collection"
    source = collection / "shared" / "owner" / "repo"
    remote = tmp_path / "upstream.git"
    _git(source, "remote", "set-url", "upstream", str(remote))
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_root", lambda: collection
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


def test_update_noop_creates_no_commit(shared_world, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    before = _git(source, "rev-parse", "HEAD").stdout.strip()

    run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(source, "branch", "--list", "ankiops/recovery/*").stdout == ""
    assert "Shared update: owner/repo — already up to date" in caplog.text
    assert "Apply to Anki: ankiops fa" in caplog.text


def test_update_reports_preserved_local_contribution(shared_world, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally committed answer"
        ),
        encoding="utf-8",
    )
    before = _commit(source, "Local contribution")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert "Shared update: owner/repo — no upstream changes" in caplog.text
    assert "Local contribution: ready to submit" in caplog.text
    assert "Apply to Anki: ankiops fa" in caplog.text
    assert "Submit contribution: ankiops shared submit owner/repo" in caplog.text


def test_submit_noop_creates_no_commit_branch_push_or_pr(shared_world, monkeypatch):
    collection, source, _remote = shared_world
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_branches = _git(source, "branch", "--format=%(refname)").stdout
    publish = monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: pytest.fail("no-op submit must not publish"),
    )
    assert publish is None

    run_submit(SimpleNamespace(repository="owner/repo", message=None))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "branch", "--format=%(refname)").stdout == before_branches
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("shared/owner/repo") is None
    finally:
        db.close()


def test_subscribe_clones_independent_repository(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    _source, remote = _setup_source(tmp_path, collection)
    shutil_source = collection / "shared" / "owner" / "repo"
    shutil.rmtree(shutil_source)
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream"):
        return original_clone(cls, str(remote_path), target, remote=remote)

    remote_path = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_root", lambda: collection
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.ensure_authenticated", lambda *_: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.repo_info",
        lambda *_: {"default_branch": "main"},
    )

    run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert GitRepository(shutil_source).is_repo()
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
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.shared.publish.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_root", lambda: collection
    )

    run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    source = collection / "shared" / "owner" / "repo"
    assert not deck.exists()
    assert GitRepository(source).is_repo()
    assert "shared/owner/repo/AnkiOpsQA" in (source / "Deck.md").read_text(
        encoding="utf-8"
    )
    assert _git(source, "branch", "--show-current").stdout.strip() == "ankiops/work"
    assert _git(remote, "rev-parse", "refs/heads/main").stdout.strip()
    db = SyncState.open(collection)
    try:
        assert db.resolve_note_sources(["create-key"])["create-key"] == (
            "shared/owner/repo"
        )
        assert db.resolve_deck_source(456) == "shared/owner/repo"
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
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push
    attempts = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def fail_once(source_git, remote_name, source_ref, branch):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(1, ["git", "push"])
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", fail_once)
    monkeypatch.setattr(
        "ankiops.shared.publish.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

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

    run_update(SimpleNamespace(repository="owner/repo"))

    assert (source / "Local.md").exists()
    assert "remote answer" in (source / "Deck.md").read_text(encoding="utf-8")
    assert (collection / "Private.md").read_bytes() == private_before
    assert (
        "Save local deck changes for owner/repo"
        in _git(source, "log", "--format=%s").stdout
    )


def test_conflict_preserves_versions_and_leaves_source_unchanged(
    shared_world, tmp_path, caplog
):
    caplog.set_level(logging.INFO)
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
        run_update(SimpleNamespace(repository="owner/repo"))

    assert deck.read_bytes() == before_deck
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    conflicts = collection / ".ankiops" / "conflicts"
    assert list(conflicts.rglob("Deck.md"))
    assert list(conflicts.rglob("Deck.md.local"))
    assert list(conflicts.rglob("Deck.md.upstream"))
    assert list(conflicts.rglob("Deck.md.base"))
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


def test_conflict_preserves_versions_for_unicode_deck_path(shared_world, tmp_path):
    collection, source, remote = shared_world
    deck_name = "Déck Ω — punctuation!.md"
    _git(source, "mv", "Deck.md", deck_name)
    _commit(source, "Rename shared deck with Unicode")
    _git(source, "push", "upstream", "HEAD:main")

    deck = source / deck_name
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "local Unicode answer"
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "unicode-upstream-edit"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Unicode Upstream")
    upstream_deck = upstream / deck_name
    upstream_deck.write_text(
        upstream_deck.read_text(encoding="utf-8").replace(
            "upstream answer", "upstream Unicode answer"
        ),
        encoding="utf-8",
    )
    _commit(upstream, "Edit Unicode deck upstream")
    _git(upstream, "push", "origin", "main")

    with pytest.raises(ValueError, match=deck_name):
        run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        conflict_root = Path(str(operation["recovery_ref"]))
    finally:
        db.close()
    assert (conflict_root / deck_name).read_text(encoding="utf-8")
    for suffix in ("base", "local", "upstream"):
        assert (conflict_root / f"{deck_name}.{suffix}").read_text(encoding="utf-8")


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
        run_update(SimpleNamespace(repository="owner/repo"))
    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: shared-key -->\nQ: shared question\nA: combined answer\n",
        encoding="utf-8",
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    assert not GitRepository(source).unmerged_paths()
    assert "combined answer" in deck.read_text(encoding="utf-8")
    assert not list((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("shared/owner/repo") is None
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
        run_submit(SimpleNamespace(repository="owner/repo", message=None))

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: shared-key -->\nQ: shared question\nA: combined\n",
        encoding="utf-8",
    )
    run_update(SimpleNamespace(repository="owner/repo"))

    assert "combined" in deck.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("shared/owner/repo") is None
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
        GitRepository,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "fetch"])
        ),
    )

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    assert deck.read_bytes() == before_bytes
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["state"] == "integrating"
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

    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
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

    args = SimpleNamespace(repository="owner/repo", message="Clarify shared answer")
    run_submit(args)
    first_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    first_refs = _git(publish, "for-each-ref", "--format=%(refname)").stdout
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == first_head
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == first_refs
    assert created_pr[0]["head"].startswith("contributor:ankiops/")
    assert _git(source, "log", "-1", "--format=%s").stdout.strip() == (
        "Save local deck changes for owner/repo"
    )
    assert private.read_text(encoding="utf-8") == "private dirty\n"
    assert _git(collection, "status", "--short").stdout.strip() == "M Private.md"
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/1"
    finally:
        db.close()


def test_submit_retry_after_failed_integration(shared_world, tmp_path, monkeypatch):
    _collection, source, _remote = shared_world
    publish = tmp_path / "integration-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "integration retry answer"
        ),
        encoding="utf-8",
    )
    original_fetch = GitRepository.fetch
    original_set_remote = GitRepository.set_remote
    fetches = 0

    def fail_once(source_git, remote_name):
        nonlocal fetches
        fetches += 1
        if fetches == 1:
            raise subprocess.CalledProcessError(1, ["git", "fetch"])
        original_fetch(source_git, remote_name)

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "fetch", fail_once)
    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/integration",
    )
    args = SimpleNamespace(repository="owner/repo", message="Retry integration")

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_submit(args)
    run_submit(args)

    assert fetches == 2
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout.strip()


def test_submit_retry_after_interruption_while_applying(
    shared_world, tmp_path, monkeypatch
):
    collection, source, _remote = shared_world
    publish = tmp_path / "apply-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "apply retry answer"
        ),
        encoding="utf-8",
    )
    original_reset_hard = GitRepository.reset_hard
    original_set_remote = GitRepository.set_remote
    interrupted = False

    def interrupt_after_reset(source_git, ref):
        nonlocal interrupted
        original_reset_hard(source_git, ref)
        if source_git.root == source and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "reset_hard", interrupt_after_reset)
    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/apply",
    )
    args = SimpleNamespace(repository="owner/repo", message="Retry apply")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["state"] == "applying"
    finally:
        db.close()

    run_submit(args)

    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout.strip()


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
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    calls = 0

    def fail_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("simulated PR failure")
        return "https://github.test/owner/repo/pull/2"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.shared.commands.GitHubHost.create_pr", fail_once)
    args = SimpleNamespace(repository="owner/repo", message="Retry PR")

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
        operation = db.get_shared_operation("shared/owner/repo")
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
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push
    pushes = 0

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    def fail_once(source_git, remote_name, source_ref, branch):
        nonlocal pushes
        pushes += 1
        if pushes == 1:
            raise subprocess.CalledProcessError(1, ["git", "push"])
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(GitRepository, "push", fail_once)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/3",
    )
    args = SimpleNamespace(repository="owner/repo", message="Retry push")

    with pytest.raises(ValueError, match="did not reach GitHub"):
        run_submit(args)
    failed_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    db = SyncState.open(collection)
    try:
        failed_operation = db.get_shared_operation("shared/owner/repo")
        assert failed_operation is not None
        branch = failed_operation["publish_branch"]
        assert failed_operation["state"] == "push_failed"
    finally:
        db.close()
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == failed_head
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["publish_branch"] == branch
        assert operation["state"] == "pr_open"
    finally:
        db.close()


def test_submit_retry_after_interruption_after_push(
    shared_world, tmp_path, monkeypatch
):
    _collection, source, _remote = shared_world
    publish = tmp_path / "pushed-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "pushed retry answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push
    pushes = 0

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    def interrupt_after_push(source_git, remote_name, source_ref, branch):
        nonlocal pushes
        pushes += 1
        original_push(source_git, remote_name, source_ref, branch)
        if pushes == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(GitRepository, "push", interrupt_after_push)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/pushed",
    )
    args = SimpleNamespace(repository="owner/repo", message="Retry pushed commit")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    run_submit(args)

    assert pushes == 1


def test_submit_retry_after_interruption_after_pr_creation(
    shared_world, tmp_path, monkeypatch
):
    collection, source, _remote = shared_world
    publish = tmp_path / "pr-created-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "PR-created retry answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote
    pr_calls = 0
    created_pr = None

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    def interrupt_after_pr(*_args, **_kwargs):
        nonlocal created_pr, pr_calls
        pr_calls += 1
        if created_pr is None:
            created_pr = "https://github.test/owner/repo/pull/existing"
            raise KeyboardInterrupt
        return created_pr

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr", interrupt_after_pr
    )
    args = SimpleNamespace(repository="owner/repo", message="Retry created PR")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    run_submit(args)

    assert pr_calls == 2
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/existing"
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
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/4",
    )
    run_submit(SimpleNamespace(repository="owner/repo", message="Merged change"))

    merged = tmp_path / "squash-merge"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged, "Maintainer")
    (merged / "Deck.md").write_bytes(deck.read_bytes())
    _commit(merged, "Squash merged contribution")
    _git(merged, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        assert db.get_shared_operation("shared/owner/repo") is None
    finally:
        db.close()
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == ""


def test_submit_auth_failure_reports_simple_retry(shared_world, monkeypatch):
    collection, source, _remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally committed answer"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.GitHubHost.publish_target",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("GitHub CLI is not authenticated. Run: gh auth login")
        ),
    )

    with pytest.raises(ValueError) as error:
        run_submit(SimpleNamespace(repository="owner/repo", message="Local change"))

    message = str(error.value)
    assert "was committed locally" in message
    assert "Nothing was uploaded" in message
    assert "Retrying is safe: ankiops shared submit owner/repo" in message
    db = SyncState.open(collection)
    try:
        operation = db.get_shared_operation("shared/owner/repo")
        assert operation is not None
        assert operation["state"] == "ready"
    finally:
        db.close()
    assert _git(source, "branch", "--list", "ankiops/*").stdout.strip() == (
        "* ankiops/work"
    )


def test_status_reports_local_state_when_fetch_fails(shared_world, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    (source / "Draft.md").write_text("draft\n", encoding="utf-8")
    monkeypatch.setattr(
        GitRepository,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "fetch"])
        ),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "Working tree: 1 changed path" in caplog.text
    assert "Upstream: unavailable" in caplog.text
    assert "collection changes" not in caplog.text.lower()
    assert "Retry status: ankiops shared status owner/repo" in caplog.text


def test_status_reports_committed_local_contribution(shared_world, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally committed answer"
        ),
        encoding="utf-8",
    )
    _commit(source, "Local contribution")

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "Shared status: owner/repo" in caplog.text
    assert "Working tree: clean" in caplog.text
    assert "Local contribution: 1 commit ready to submit" in caplog.text
    assert "Upstream: up to date" in caplog.text
    assert "Anki: changes not applied" in caplog.text
    assert "Apply to Anki: ankiops fa" in caplog.text
    assert "Submit contribution: ankiops shared submit owner/repo" in caplog.text


def test_status_reports_available_upstream_update(shared_world, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _collection, _source, remote = shared_world
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "Local contribution: none" in caplog.text
    assert "Upstream: 1 commit available" in caplog.text
    assert "Integrate upstream: ankiops shared update owner/repo" in caplog.text
    assert "Submit contribution:" not in caplog.text


def test_status_reports_diverged_contributions(shared_world, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, remote = shared_world
    (source / "Local.md").write_text("local\n", encoding="utf-8")
    _commit(source, "Local contribution")
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "Local contribution: 1 commit ready to submit" in caplog.text
    assert "Upstream: 1 commit available; update before submitting" in caplog.text
    assert "Integrate upstream: ankiops shared update owner/repo" in caplog.text
    assert "Submit contribution:" not in caplog.text


def test_status_reports_contribution_accepted_upstream(shared_world, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    collection, source, remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "accepted answer"),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Local contribution")
    upstream_tree = _git(source, "rev-parse", "upstream/main^{tree}").stdout.strip()
    _upstream_edit(tmp_path, remote, "upstream answer", "accepted answer")
    db = SyncState.open(collection)
    try:
        db.save_shared_operation(
            "shared/owner/repo",
            "accepted-operation",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=upstream_tree,
            publish_branch="ankiops/accepted-operation",
            pr_url="https://github.test/owner/repo/pull/1",
        )
    finally:
        db.close()

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "Local contribution: none" in caplog.text
    assert "Upstream: up to date" in caplog.text
    assert "Submission: accepted upstream; cleanup pending" in caplog.text
    assert "Finalize submission: ankiops shared update owner/repo" in caplog.text
    assert "Review pull request:" not in caplog.text


def test_status_omits_unrelated_collection_changes(shared_world, caplog):
    caplog.set_level(logging.INFO)
    collection, _source, _remote = shared_world
    private_deck = collection / "Private.md"
    private_deck.write_text("private edit\n", encoding="utf-8")
    generated_config = collection / "note_types" / "Generated.yaml"
    generated_config.parent.mkdir()
    generated_config.write_text("generated\n", encoding="utf-8")

    run_status(SimpleNamespace(repository="owner/repo"))

    assert "collection changes" not in caplog.text.lower()
    assert "Private.md" not in caplog.text
    assert "Generated.yaml" not in caplog.text


def test_status_prints_unicode_shared_paths_without_git_quoting(shared_world, caplog):
    caplog.set_level(logging.INFO)
    _collection, source, _remote = shared_world
    unicode_name = "Déck Ω — punctuation!.md"
    (source / unicode_name).write_text("draft\n", encoding="utf-8")

    run_status(SimpleNamespace(repository="owner/repo"))

    assert unicode_name in caplog.text
    assert "\\303" not in caplog.text


@pytest.mark.parametrize(
    ("state", "submission_status"),
    [
        ("push_failed", "not uploaded"),
        ("pr_failed", "uploaded; pull request not created"),
    ],
)
def test_status_gives_submit_retry_as_next_command(
    shared_world, caplog, state, submission_status
):
    caplog.set_level(logging.INFO)
    collection, source, _remote = shared_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "local contribution"
        ),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Local contribution")
    db = SyncState.open(collection)
    try:
        db.save_shared_operation(
            "shared/owner/repo",
            "retry-operation",
            "submit",
            state,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            prepared_head=prepared_head,
            publish_branch="ankiops/retry-operation",
        )
    finally:
        db.close()

    run_status(SimpleNamespace(repository="owner/repo"))

    assert f"Submission: {submission_status}" in caplog.text
    assert "Retry submission: ankiops shared submit owner/repo" in caplog.text

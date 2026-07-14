from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.collab.commands import (
    run_publish,
    run_status,
    run_submit,
    run_subscribe,
    run_update,
)
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
    _git(source, "update-ref", "refs/ankiops/integrated", "upstream/main")
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


def test_update_noop_creates_no_commit(collab_world, capsys):
    _collection, source, _remote = collab_world
    before = _git(source, "rev-parse", "HEAD").stdout.strip()

    run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(source, "branch", "--list", "ankiops/recovery/*").stdout == ""
    output = capsys.readouterr().out
    assert "✓ Update owner/repo" in output
    assert "already up to date" in output
    assert "Apply to Anki" not in output


def test_update_separates_multiple_source_blocks(collab_world, capsys):
    collection, _source, remote = collab_world
    second = collection / "collab" / "second-owner" / "second-repo"
    second.parent.mkdir(parents=True)
    _git(second.parent, "clone", "--origin", "upstream", str(remote), str(second))
    _configure(second, "Second Contributor")
    _git(second, "checkout", "-b", "ankiops/journal", "upstream/main")
    _git(second, "branch", "--unset-upstream")
    _git(second, "update-ref", "refs/ankiops/integrated", "upstream/main")

    run_update(SimpleNamespace(repository=None))

    output = capsys.readouterr().out
    assert "already up to date\n\n✓ Update second-owner/second-repo" in output
    assert output.count("already up to date") == 2


def test_update_all_continues_after_one_source_fails(collab_world, monkeypatch):
    collection, _source, remote = collab_world
    second = collection / "collab" / "second-owner" / "second-repo"
    second.parent.mkdir(parents=True)
    _git(second.parent, "clone", "--origin", "upstream", str(remote), str(second))
    _configure(second, "Second Contributor")
    _git(second, "checkout", "-b", "ankiops/journal", "upstream/main")
    _git(second, "branch", "--unset-upstream")
    _git(second, "update-ref", "refs/ankiops/integrated", "upstream/main")
    attempted = []

    def update_one(_root, source, _state):
        attempted.append(source.display_name)
        if source.display_name == "owner/repo":
            raise ValueError("simulated first-source failure")

    monkeypatch.setattr("ankiops.collab.commands._update_one", update_one)

    with pytest.raises(
        ValueError,
        match="1 failure.*owner/repo: simulated first-source failure",
    ):
        run_update(SimpleNamespace(repository=None))

    assert attempted == ["owner/repo", "second-owner/second-repo"]


def test_update_reports_preserved_local_contribution(collab_world, capsys):
    _collection, source, _remote = collab_world
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
    output = capsys.readouterr().out
    assert "✓ Update owner/repo" in output
    assert "no upstream changes" in output
    assert "Local contribution: ready to submit" in output
    assert "Apply to Anki" not in output
    assert "Submit contribution: ankiops collab submit owner/repo" in output


def test_submit_noop_creates_no_commit_branch_push_or_pr(collab_world, monkeypatch):
    collection, source, _remote = collab_world
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_branches = _git(source, "branch", "--format=%(refname)").stdout
    publish = monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: pytest.fail("no-op submit must not publish"),
    )
    assert publish is None

    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "branch", "--format=%(refname)").stdout == before_branches
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_submit_rejects_duplicate_note_keys_before_github_with_both_locations(
    collab_world, monkeypatch
):
    collection, source, _remote = collab_world
    (source / "Second.md").write_text(
        "<!-- note_key: collab-key -->\nQ: duplicate\nA: answer\n",
        encoding="utf-8",
    )
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_status = _git(source, "status", "--short").stdout
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: pytest.fail("duplicate keys must fail before GitHub setup"),
    )

    with pytest.raises(ValueError) as raised:
        run_submit(SimpleNamespace(repository="owner/repo", title=None))

    message = str(raised.value)
    assert "Duplicate note_key 'collab-key'" in message
    assert "Deck.md note 1" in message
    assert "Second.md note 1" in message
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--short").stdout == before_status
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_subscribe_clones_independent_repository(tmp_path, monkeypatch, capsys):
    collection = _setup_collection(tmp_path)
    _source, remote = _setup_source(tmp_path, collection)
    shutil_source = collection / "collab" / "owner" / "repo"
    shutil.rmtree(shutil_source)
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        assert anonymous is True
        return original_clone(cls, str(remote_path), target, remote=remote)

    remote_path = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.ensure_authenticated",
        lambda *_: pytest.fail("public subscribe must not require GitHub auth"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.repo_info",
        lambda *_: pytest.fail("public subscribe must clone directly"),
    )

    run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert GitRepository(shutil_source).is_repo()
    assert _git(shutil_source, "branch", "--show-current").stdout.strip() == (
        "ankiops/journal"
    )
    assert (
        _git(shutil_source, "rev-parse", "refs/ankiops/integrated").stdout.strip()
        == _git(shutil_source, "rev-parse", "upstream/main").stdout.strip()
    )
    assert (
        _git(
            shutil_source,
            "config",
            "--get",
            "branch.ankiops/journal.remote",
            check=False,
        ).returncode
        == 1
    )
    assert _git(shutil_source, "config", "user.name").stdout.strip() == "Private User"
    assert _git(collection, "status", "--short").stdout == ""
    output = capsys.readouterr().out
    assert "✓ Subscribe owner/repo" in output
    assert "ankiops fa" in output


def test_subscribe_interruption_removes_the_partial_repository(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    source = collection / "collab" / "owner" / "repo"
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    def interrupted_clone(_cls, _url, target, **_kwargs):
        target.mkdir(parents=True)
        (target / "partial-clone").write_text("incomplete\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "clone", classmethod(interrupted_clone))

    with pytest.raises(KeyboardInterrupt):
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    assert not source.exists()


def test_subscribe_rejects_duplicate_note_keys_with_both_locations(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    source, remote = _setup_source(tmp_path, collection)
    (source / "Second.md").write_text(
        "<!-- note_key: collab-key -->\nQ: duplicate\nA: answer\n",
        encoding="utf-8",
    )
    _commit(source, "Add duplicate note key")
    _git(source, "push", "upstream", "HEAD:main")
    shutil.rmtree(collection / "collab")
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        return original_clone(cls, str(remote_path), target, remote=remote)

    remote_path = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError) as raised:
        run_subscribe(SimpleNamespace(repository="owner/repo"))

    message = str(raised.value)
    assert "Duplicate note_key 'collab-key'" in message
    assert "Deck.md note 1" in message
    assert "Second.md note 1" in message
    assert not (collection / "collab" / "owner" / "repo").exists()


def test_subscribe_explains_when_a_fork_belongs_to_the_upstream_identity(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    source, remote = _setup_source(tmp_path, collection)
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "<!-- note_key: collab-key -->",
            "<!-- note_key: collab-key -->\n"
            "<!-- note_type: collab/canonical_owner/QA_/AnkiOpsQA -->",
        ),
        encoding="utf-8",
    )
    _commit(source, "Bind deck to canonical collab identity")
    _git(source, "push", "upstream", "HEAD:main")
    shutil.rmtree(collection / "collab")
    original_clone = GitRepository.clone.__func__

    def local_clone(cls, _url, target, *, remote="upstream", anonymous=False):
        assert anonymous is True
        return original_clone(cls, str(remote_path), target, remote=remote)

    remote_path = remote
    monkeypatch.setattr(GitRepository, "clone", classmethod(local_clone))
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(
        ValueError,
        match=(
            "belong to canonical_owner/QA_.*"
            "ankiops collab subscribe canonical_owner/QA_"
        ),
    ):
        run_subscribe(SimpleNamespace(repository="fork-owner/fork_repo"))

    assert not (collection / "collab" / "fork-owner" / "fork_repo").exists()


def test_publish_moves_deck_into_independent_repository(tmp_path, monkeypatch, capsys):
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
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    source = collection / "collab" / "owner" / "repo"
    assert not deck.exists()
    assert GitRepository(source).is_repo()
    assert "collab/owner/repo/AnkiOpsQA" in (source / "Deck.md").read_text(
        encoding="utf-8"
    )
    readme = (source / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Deck\n")
    assert "ankiops collab subscribe owner/repo" in readme
    assert "GitHub CLI" in readme
    assert "installed and authenticated" in readme
    assert readme.count("files-to-anki") == 1
    assert readme.count("anki-to-files") == 1
    assert "{{" not in readme
    assert _git(remote, "show", "main:README.md").stdout == readme
    assert _git(source, "branch", "--show-current").stdout.strip() == (
        "ankiops/journal"
    )
    assert _git(remote, "rev-parse", "refs/heads/main").stdout.strip()
    db = SyncState.open(collection)
    try:
        assert db.resolve_note_sources(["create-key"])["create-key"] == (
            "collab/owner/repo"
        )
        assert db.resolve_deck_source(456) == "collab/owner/repo"
    finally:
        db.close()
    output = capsys.readouterr().out
    assert output.count("Apply to Anki: ankiops fa") == 1


def test_publish_exports_only_the_referenced_note_type_manifest(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    private = collection / "PrivateSecret.txt"
    private.write_text("root secret\n", encoding="utf-8")
    qa_dir = collection / "note_types" / "AnkiOpsQA"
    (qa_dir / "UnreferencedSecret.txt").write_text(
        "note-type secret\n", encoding="utf-8"
    )
    (qa_dir / "PrivateLink.txt").symlink_to(private)
    (collection / "Deck.md").write_text(
        "<!-- note_key: manifest-key -->\nQ: shared\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add deck with private note-type auxiliaries")
    remote = tmp_path / "manifest.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_publish(SimpleNamespace(deck="Deck", repository="owner/manifest"))

    public_paths = set(
        _git(remote, "ls-tree", "-r", "--name-only", "main").stdout.splitlines()
    )
    public_paths.discard(".gitignore")
    assert public_paths == {
        "Deck.md",
        "README.md",
        "note_types/AnkiOpsQA/Back.template.anki",
        "note_types/AnkiOpsQA/Front.template.anki",
        "note_types/AnkiOpsQA/note_type.yaml",
        "note_types/AnkiOpsStyling.css",
        "note_types/SyntaxHighlighting.css",
    }
    assert private.read_text(encoding="utf-8") == "root secret\n"


def test_publish_rejects_a_symlinked_selected_deck(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    private = collection / "PrivateSource.md"
    private.write_text(
        "<!-- note_key: private-source-key -->\nQ: private\nA: secret\n",
        encoding="utf-8",
    )
    (collection / "Shared.md").symlink_to(private)
    _commit(collection, "Add symlinked deck")
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "symlinked deck must fail before creating a GitHub repository"
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_publish(SimpleNamespace(deck="Shared", repository="owner/symlink-deck"))

    assert private.read_text(encoding="utf-8").endswith("A: secret\n")
    assert not (collection / "collab" / "owner" / "symlink-deck").exists()


def test_publish_rejects_a_referenced_symlinked_note_type_asset(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    qa_dir = collection / "note_types" / "AnkiOpsQA"
    private = collection / "PrivateTemplate.txt"
    private.write_text("{{Question}}\n", encoding="utf-8")
    front = qa_dir / "Front.template.anki"
    front.unlink()
    front.symlink_to(private)
    (collection / "Shared.md").write_text(
        "<!-- note_key: symlink-template-key -->\nQ: shared\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add symlinked note-type template")
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "symlinked asset must fail before creating a GitHub repository"
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="outside the note_types|symbolic link"):
        run_publish(SimpleNamespace(deck="Shared", repository="owner/symlink-template"))

    assert private.read_text(encoding="utf-8") == "{{Question}}\n"
    assert not (collection / "collab" / "owner" / "symlink-template").exists()


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
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_publish(args)
    source = collection / "collab" / "owner" / "repo"
    first_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    run_publish(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == first_head
    assert _git(source, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert not deck.exists()


def test_publish_retry_rejects_a_referenced_asset_changed_after_preparation(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: retry-asset-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    styling = collection / "note_types" / "AnkiOpsStyling.css"
    _commit(collection, "Add retry-asset deck")
    remote = tmp_path / "retry-asset.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    push_attempts = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def fail_push(_source_git, _remote_name, _source_ref, _branch):
        nonlocal push_attempts
        push_attempts += 1
        raise subprocess.CalledProcessError(1, ["git", "push"])

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", fail_push)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_publish(args)
    styling.write_bytes(styling.read_bytes() + b"\n/* newer local edit */\n")

    with pytest.raises(ValueError, match="publish-applicable files changed since"):
        run_publish(args)

    assert push_attempts == 1
    assert deck.exists()


def test_publish_aborts_if_a_selected_root_deck_changes_during_github_work(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    original_content = (
        "<!-- note_key: concurrent-publish-key -->\nQ: original\nA: answer\n"
    )
    concurrent_content = (
        "<!-- note_key: concurrent-publish-key -->\nQ: edited while publishing\n"
        "A: must survive\n"
    )
    deck.write_text(original_content, encoding="utf-8")
    _commit(collection, "Add publish deck")
    remote = tmp_path / "concurrent-publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def edit_during_push(source_git, remote_name, source_ref, branch):
        deck.write_text(concurrent_content, encoding="utf-8")
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", edit_during_push)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="selected local deck files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.read_text(encoding="utf-8") == concurrent_content
    published_content = _git(remote, "show", "main:Deck.md").stdout
    assert "Q: original" in published_content
    assert "edited while publishing" not in published_content

    with pytest.raises(ValueError, match="local deck files changed since"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.read_text(encoding="utf-8") == concurrent_content
    assert "edited while publishing" not in _git(remote, "show", "main:Deck.md").stdout


def test_publish_aborts_if_a_referenced_note_type_asset_changes_during_github_work(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: concurrent-asset-key -->\nQ: original\nA: answer\n",
        encoding="utf-8",
    )
    styling = collection / "note_types" / "AnkiOpsStyling.css"
    original_styling = styling.read_bytes()
    concurrent_styling = original_styling + b"\n/* concurrent edit */\n"
    _commit(collection, "Add publish deck")
    remote = tmp_path / "concurrent-asset-publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def edit_during_push(source_git, remote_name, source_ref, branch):
        styling.write_bytes(concurrent_styling)
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", edit_during_push)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.exists()
    assert styling.read_bytes() == concurrent_styling
    assert (
        _git(remote, "show", "main:note_types/AnkiOpsStyling.css").stdout.encode()
        == original_styling
    )


def test_publish_aborts_if_a_selected_subdeck_is_added_during_github_work(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: concurrent-add-key -->\nQ: original\nA: answer\n",
        encoding="utf-8",
    )
    added = collection / "Deck__Added.md"
    added_content = "<!-- note_key: added-key -->\nQ: added\nA: must survive\n"
    _commit(collection, "Add publish deck")
    remote = tmp_path / "concurrent-add-publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def add_during_push(source_git, remote_name, source_ref, branch):
        added.write_text(added_content, encoding="utf-8")
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", add_during_push)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.exists()
    assert added.read_text(encoding="utf-8") == added_content
    assert _git(remote, "show", "main:Deck__Added.md", check=False).returncode != 0


def test_publish_aborts_if_referenced_media_is_deleted_during_github_work(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    media = collection / "media" / "shared.png"
    media.parent.mkdir()
    original_media = b"original media bytes\x00"
    media.write_bytes(original_media)
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: concurrent-delete-key -->\n"
        "Q: original\n"
        "A: ![shared](media/shared.png)\n",
        encoding="utf-8",
    )
    _commit(collection, "Add publish deck with media")
    remote = tmp_path / "concurrent-delete-publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def delete_during_push(source_git, remote_name, source_ref, branch):
        media.unlink()
        original_push(source_git, remote_name, source_ref, branch)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "push", delete_during_push)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.exists()
    assert not media.exists()
    published_media = _git(remote, "show", "main:media/shared.png").stdout.encode(
        errors="surrogateescape"
    )
    assert published_media == original_media


def test_publish_never_removes_a_deck_edited_at_the_final_handoff(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: final-handoff-key -->\nQ: original\nA: answer\n",
        encoding="utf-8",
    )
    concurrent_content = (
        "<!-- note_key: final-handoff-key -->\nQ: newest edit\nA: must survive\n"
    )
    _commit(collection, "Add final-handoff deck")
    remote = tmp_path / "final-handoff.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    from ankiops.collab import publish as publish_module

    original_remove = publish_module._remove_published_local_files

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def edit_at_handoff(plan):
        deck.write_text(concurrent_content, encoding="utf-8")
        original_remove(plan)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.publish._remove_published_local_files", edit_at_handoff
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.read_text(encoding="utf-8") == concurrent_content


def test_publish_rechecks_referenced_assets_before_ownership_transfer(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    original_deck = "<!-- note_key: ownership-race-key -->\nQ: original\nA: answer\n"
    deck.write_text(original_deck, encoding="utf-8")
    styling = collection / "note_types" / "AnkiOpsStyling.css"
    concurrent_styling = styling.read_bytes() + b"\n/* ownership race */\n"
    _commit(collection, "Add ownership-race deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("ownership-race-key", 123)])
    db.close()
    remote = tmp_path / "ownership-race.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_commit_paths = GitRepository.commit_paths

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def edit_after_root_commit(repository, paths, message):
        result = original_commit_paths(repository, paths, message)
        styling.write_bytes(concurrent_styling)
        return result

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(GitRepository, "commit_paths", edit_after_root_commit)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original_deck
    assert styling.read_bytes() == concurrent_styling
    db = SyncState.open(collection)
    try:
        assert db.resolve_note_sources(["ownership-race-key"])[
            "ownership-race-key"
        ] == (".")
    finally:
        db.close()


def test_publish_writes_prepared_assets_from_the_immutable_manifest(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: immutable-manifest-key -->\nQ: original\nA: answer\n",
        encoding="utf-8",
    )
    styling = collection / "note_types" / "AnkiOpsStyling.css"
    original_styling = styling.read_bytes()
    concurrent_styling = original_styling + b"\n/* edit after snapshot */\n"
    _commit(collection, "Add immutable-manifest deck")
    remote = tmp_path / "immutable-manifest.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote

    class EditingHost:
        def __init__(self):
            styling.write_bytes(concurrent_styling)

        def create_repo(self, _slug):
            return None

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost", lambda _root: EditingHost()
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="publish-applicable files changed"):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    source_styling = (
        collection / "collab" / "owner" / "repo" / "note_types" / "AnkiOpsStyling.css"
    )
    assert styling.read_bytes() == concurrent_styling
    assert source_styling.read_bytes() == original_styling
    assert (
        _git(remote, "show", "main:note_types/AnkiOpsStyling.css").stdout.encode()
        == original_styling
    )


def test_publish_retry_rejects_a_tampered_prepared_source(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: tamper-key -->\nQ: intended\nA: deck\n",
        encoding="utf-8",
    )
    _commit(collection, "Add retry deck")
    remote = tmp_path / "tamper.git"
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
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_publish(args)
    source = collection / "collab" / "owner" / "repo"
    private = source / "PrivateSecret.txt"
    private.write_text("must not be uploaded\n", encoding="utf-8")
    _commit(source, "Add unrelated private content")

    with pytest.raises(ValueError, match="interrupted publish source changed"):
        run_publish(args)

    assert deck.exists()
    assert private.exists()
    assert _git(remote, "show", "main:PrivateSecret.txt", check=False).returncode != 0


def test_publish_retry_after_repository_creation_interruption(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: retry-created-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add retry-after-create deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("retry-created-key", 123)])
    db.close()
    remote = tmp_path / "created-after-interruption.git"
    original_set_remote = GitRepository.set_remote
    create_calls = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def create_then_interrupt(*_args, **_kwargs):
        nonlocal create_calls
        create_calls += 1
        _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
        raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", create_then_interrupt
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.repo_info",
        lambda *_args: {"full_name": "owner/repo"} if remote.exists() else None,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(KeyboardInterrupt):
        run_publish(args)
    source = collection / "collab" / "owner" / "repo"
    assert GitRepository(source).is_repo()

    run_publish(args)

    assert create_calls == 1
    assert not deck.exists()
    assert _git(remote, "show", "main:Deck.md").stdout
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
        assert (
            db.resolve_note_sources(["retry-created-key"])["retry-created-key"]
            == "collab/owner/repo"
        )
    finally:
        db.close()


def test_publish_preserves_source_and_adopts_empty_remote_after_ambiguous_create(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: ambiguous-create-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add ambiguous-create deck")
    remote = tmp_path / "ambiguous-create.git"
    original_set_remote = GitRepository.set_remote
    create_calls = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def ambiguous_gh(_host, args, *, check=True):
        nonlocal create_calls
        if args[:2] == ["auth", "status"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:2] == ["api", "repos/owner/repo"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")
        if args[:2] == ["repo", "create"]:
            create_calls += 1
            _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr="connection closed after request",
            )
        raise AssertionError(f"Unexpected gh call: {args}")

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr("ankiops.collab.publish.GitHubHost._gh", ambiguous_gh)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="connection closed after request"):
        run_publish(args)

    source = collection / "collab" / "owner" / "repo"
    assert GitRepository(source).is_repo()
    prepared_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    assert (source / "Deck.md").exists()
    assert deck.exists()

    run_publish(args)

    assert create_calls == 1
    assert not deck.exists()
    assert _git(remote, "rev-parse", "main").stdout.strip() == prepared_head


def test_publish_retry_after_local_removal_before_ownership_transfer(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: retry-transfer-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add transfer-retry deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("retry-transfer-key", 321)])
    db.upsert_deck("Deck", 654)
    db.close()
    remote = tmp_path / "transfer-retry.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    from ankiops.collab import publish as publish_module

    original_transfer = publish_module._transfer_sync_ownership
    transfer_calls = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def interrupt_once(collection_root, plan):
        nonlocal transfer_calls
        transfer_calls += 1
        if transfer_calls == 1:
            raise KeyboardInterrupt
        original_transfer(collection_root, plan)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.publish._transfer_sync_ownership", interrupt_once
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(KeyboardInterrupt):
        run_publish(args)
    assert not deck.exists()

    run_publish(args)

    assert transfer_calls == 2
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
        assert (
            db.resolve_note_sources(["retry-transfer-key"])["retry-transfer-key"]
            == "collab/owner/repo"
        )
        assert db.resolve_deck_source(654) == "collab/owner/repo"
    finally:
        db.close()


def test_publish_existing_remote_is_terminal_and_preserves_local_state(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    original_content = (
        "<!-- note_key: existing-remote-key -->\nQ: local\nA: must survive\n"
    )
    deck.write_text(original_content, encoding="utf-8")
    _commit(collection, "Add collision deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("existing-remote-key", 123)])
    db.upsert_deck("Deck", 456)
    db.close()
    monkeypatch.setattr(
        "ankiops.collab.hosting.GitHubHost.ensure_authenticated", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.hosting.GitHubHost.repo_info",
        lambda *_args: {"full_name": "owner/repo"},
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError) as raised:
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    message = str(raised.value)
    assert "already exists" in message
    assert "Choose a new name" in message
    assert "Retrying is safe" not in message
    assert deck.read_text(encoding="utf-8") == original_content
    assert not (collection / "collab" / "owner" / "repo").exists()
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
        assert (
            db.resolve_note_sources(["existing-remote-key"])["existing-remote-key"]
            == "."
        )
        assert db.resolve_deck_source(456) == "."
    finally:
        db.close()


def test_publish_retry_rejects_an_unrelated_remote_created_during_interruption(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: unrelated-remote-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add collision-test deck")
    remote = tmp_path / "unrelated.git"
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def create_unrelated_then_interrupt(*_args, **_kwargs):
        _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
        unrelated = tmp_path / "unrelated-seed"
        _git(tmp_path, "clone", str(remote), str(unrelated))
        _configure(unrelated, "Unrelated Owner")
        (unrelated / "Unrelated.md").write_text("unrelated\n", encoding="utf-8")
        _commit(unrelated, "Unrelated repository")
        _git(unrelated, "push", "origin", "main")
        raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        create_unrelated_then_interrupt,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(KeyboardInterrupt):
        run_publish(args)
    source = collection / "collab" / "owner" / "repo"
    assert source.exists()
    unrelated_head = _git(remote, "rev-parse", "main").stdout.strip()

    with pytest.raises(ValueError) as raised:
        run_publish(args)

    message = str(raised.value)
    assert "unrelated content" in message
    assert "Choose a new name" in message
    assert "Retrying is safe" not in message
    assert deck.exists()
    assert not source.exists()
    assert _git(remote, "rev-parse", "main").stdout.strip() == unrelated_head
    assert _git(remote, "show", "main:Unrelated.md").stdout == "unrelated\n"
    assert _git(remote, "show", "main:Deck.md", check=False).returncode != 0
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_publish_retry_rejects_a_remote_with_history_only_on_another_branch(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: other-branch-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add other-branch collision deck")
    remote = tmp_path / "other-branch.git"
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def create_other_branch_then_interrupt(*_args, **_kwargs):
        _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
        unrelated = tmp_path / "other-branch-seed"
        unrelated.mkdir()
        _git(unrelated, "init", "-b", "main")
        _configure(unrelated, "Unrelated Owner")
        (unrelated / "Unrelated.md").write_text("unrelated\n", encoding="utf-8")
        unrelated_head = _commit(unrelated, "Unrelated repository")
        _git(unrelated, "remote", "add", "origin", str(remote))
        _git(unrelated, "push", "origin", f"{unrelated_head}:refs/heads/unrelated")
        raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        create_other_branch_then_interrupt,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(KeyboardInterrupt):
        run_publish(args)

    with pytest.raises(ValueError, match="unrelated content"):
        run_publish(args)

    assert deck.exists()
    assert _git(remote, "show", "main:Deck.md", check=False).returncode != 0
    assert _git(remote, "show", "unrelated:Unrelated.md").stdout == "unrelated\n"


def test_publish_collision_preserves_prepared_source_when_root_copy_is_gone(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    original_content = (
        "<!-- note_key: only-prepared-copy-key -->\nQ: only copy\nA: preserve me\n"
    )
    deck.write_text(original_content, encoding="utf-8")
    _commit(collection, "Add only-copy collision deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("only-prepared-copy-key", 987)])
    db.close()
    remote = tmp_path / "only-copy-collision.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def interrupt_before_ownership_transfer(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.publish._transfer_sync_ownership",
        interrupt_before_ownership_transfer,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(KeyboardInterrupt):
        run_publish(args)

    source = collection / "collab" / "owner" / "repo"
    prepared_deck = source / "Deck.md"
    assert not deck.exists()
    prepared_content = prepared_deck.read_bytes()
    assert b"only-prepared-copy-key" in prepared_content
    assert b"preserve me" in prepared_content

    unrelated = tmp_path / "replacement-history"
    unrelated.mkdir()
    _git(unrelated, "init", "-b", "main")
    _configure(unrelated, "Unrelated Owner")
    (unrelated / "Unrelated.md").write_text("unrelated\n", encoding="utf-8")
    _commit(unrelated, "Replace remote history")
    _git(unrelated, "remote", "add", "origin", str(remote))
    _git(unrelated, "push", "--force", "origin", "main")
    unrelated_head = _git(remote, "rev-parse", "main").stdout.strip()

    with pytest.raises(ValueError) as raised:
        run_publish(args)

    message = str(raised.value)
    assert "unrelated content" in message
    assert "Choose a new name" in message
    assert "Retrying is safe" not in message
    assert not deck.exists()
    assert prepared_deck.read_bytes() == prepared_content
    assert _git(remote, "rev-parse", "main").stdout.strip() == unrelated_head
    assert _git(remote, "show", "main:Deck.md", check=False).returncode != 0
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
        assert (
            db.resolve_note_sources(["only-prepared-copy-key"])[
                "only-prepared-copy-key"
            ]
            == "."
        )
    finally:
        db.close()


def test_publish_ownership_transfer_rolls_back_as_one_transaction(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: atomic-transfer-key -->\nQ: retry\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add atomic-transfer deck")
    db = SyncState.open(collection)
    db.upsert_note_links([("atomic-transfer-key", 321)])
    db.upsert_deck("Deck", 654)
    db.close()
    remote = tmp_path / "atomic-transfer.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote
    original_upsert_deck = SyncState.upsert_deck
    transfer_attempts = 0

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    def fail_once(self, *args, **kwargs):
        nonlocal transfer_attempts
        transfer_attempts += 1
        if transfer_attempts == 1:
            raise RuntimeError("interrupt ownership transfer")
        original_upsert_deck(self, *args, **kwargs)

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(SyncState, "upsert_deck", fail_once)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    with pytest.raises(ValueError, match="interrupt ownership transfer"):
        run_publish(args)
    db = SyncState.open(collection)
    try:
        assert (
            db.resolve_note_sources(["atomic-transfer-key"])["atomic-transfer-key"]
            == "."
        )
        assert db.resolve_deck_source(654) == "."
    finally:
        db.close()

    run_publish(args)

    db = SyncState.open(collection)
    try:
        assert (
            db.resolve_note_sources(["atomic-transfer-key"])["atomic-transfer-key"]
            == "collab/owner/repo"
        )
        assert db.resolve_deck_source(654) == "collab/owner/repo"
    finally:
        db.close()


def test_publish_rejects_reusing_repository_for_a_different_deck(tmp_path, monkeypatch):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    first = collection / "First.md"
    first.write_text(
        "<!-- note_key: first-key -->\nQ: first\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add first deck")
    remote = tmp_path / "published.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    original_set_remote = GitRepository.set_remote

    def local_remote(source_git, name, _url):
        original_set_remote(source_git, name, str(remote))

    monkeypatch.setattr(GitRepository, "set_remote", local_remote)
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(deck="First", repository="owner/repo")
    run_publish(args)

    second = collection / "Second.md"
    second.write_text(
        "<!-- note_key: second-key -->\nQ: second\nA: answer\n",
        encoding="utf-8",
    )
    _commit(collection, "Add second deck")

    with pytest.raises(ValueError, match="already exists"):
        run_publish(SimpleNamespace(deck="Second", repository="owner/repo"))

    assert second.exists()
    assert not (collection / "collab" / "owner" / "repo" / "Second.md").exists()
    assert _git(remote, "show", "main:Second.md", check=False).returncode != 0


def test_publish_retry_does_not_adopt_a_preexisting_local_repository(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck.md"
    deck.write_text(
        "<!-- note_key: collision-key -->\nQ: intended\nA: deck\n",
        encoding="utf-8",
    )
    _commit(collection, "Add intended deck")
    source = collection / "collab" / "owner" / "repo"
    source.mkdir(parents=True)
    _git(source, "init", "-b", "main")
    _configure(source, "Unrelated Owner")
    unrelated = source / "Unrelated.md"
    unrelated.write_text("private unrelated content\n", encoding="utf-8")
    _commit(source, "Unrelated local repository")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "a pre-existing local repository must never reach GitHub setup"
        ),
    )
    args = SimpleNamespace(deck="Deck", repository="owner/repo")

    for _attempt in range(2):
        with pytest.raises(ValueError, match="already exists") as raised:
            run_publish(args)
        assert "Retrying is safe" not in str(raised.value)

    assert deck.exists()
    assert unrelated.read_text(encoding="utf-8") == "private unrelated content\n"
    assert _git(source, "status", "--short").stdout == ""
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_status_shows_an_early_failed_publish_and_its_exact_retry(
    tmp_path, monkeypatch, capsys
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    deck = collection / "Deck with spaces.md"
    deck.write_text("Q: missing key\nA: answer\n", encoding="utf-8")
    _commit(collection, "Add invalid publish deck")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    args = SimpleNamespace(
        deck="Deck with spaces",
        repository="owner/repo",
    )

    with pytest.raises(ValueError, match="note_key"):
        run_publish(args)
    capsys.readouterr()

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert output.startswith("owner/repo\n")
    assert "Publish: failed" in output
    assert "Local repository: not prepared" in output
    assert (
        "Retry publish: ankiops collab publish 'Deck with spaces' owner/repo" in output
    )


def test_unfiltered_status_lists_publish_operations_without_source_directories(
    tmp_path, monkeypatch, capsys
):
    collection = _setup_collection(tmp_path)
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/alpha/pending",
            "pending-publish",
            "publish",
            "publishing",
            recovery_ref="ankiops collab publish Alpha alpha/pending",
        )
        db.save_collab_operation(
            "collab/zeta/failed",
            "failed-publish",
            "publish",
            "failed",
            recovery_ref="ankiops collab publish Zeta zeta/failed",
            last_error="GitHub unavailable",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    run_status(SimpleNamespace(repository=None))

    output = capsys.readouterr().out
    assert output.startswith("alpha/pending\n")
    assert "Publish: publishing" in output
    assert "zeta/failed\n  Publish: failed" in output
    assert "Last error: GitHub unavailable" in output
    assert "No subscribed collab decks" not in output


def test_publish_retry_clears_the_previous_error_when_the_new_attempt_starts(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "retry-publish",
            "publish",
            "failed",
            recovery_ref="ankiops collab publish Deck owner/repo",
            last_error="old failure",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.publish_collab_deck",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        run_publish(SimpleNamespace(deck="Deck", repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "publishing"
        assert operation["last_error"] is None
    finally:
        db.close()


@pytest.mark.parametrize(
    "media_reference",
    [
        "media/%2e%2e/PrivateSecret.txt",
        "media/../PrivateSecret.txt",
        r"media\..\PrivateSecret.txt",
    ],
    ids=["url-encoded-parent", "literal-parent", "backslash-parent"],
)
def test_publish_rejects_media_paths_that_escape_the_media_directory(
    tmp_path, monkeypatch, media_reference
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    (collection / "media").mkdir()
    private_file = collection / "PrivateSecret.txt"
    private_file.write_text("must never be published\n", encoding="utf-8")
    deck = collection / "Shared.md"
    deck.write_text(
        "<!-- note_key: traversal-key -->\n"
        "Q: shared\n"
        f"A: ![secret]({media_reference})\n",
        encoding="utf-8",
    )
    _commit(collection, "Add traversal attempt")
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid media must fail before creating a GitHub repository"
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="media path.*outside.*media"):
        run_publish(SimpleNamespace(deck="Shared", repository="owner/traversal"))

    assert private_file.read_text(encoding="utf-8") == "must never be published\n"
    assert not (collection / "collab" / "owner" / "traversal").exists()


@pytest.mark.parametrize("escape_kind", ["absolute", "symlink"])
def test_publish_rejects_absolute_and_symlink_media_escapes(
    tmp_path, monkeypatch, escape_kind
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    media = collection / "media"
    media.mkdir()
    private_file = collection / "PrivateSecret.txt"
    private_file.write_text("must never be published\n", encoding="utf-8")
    if escape_kind == "absolute":
        media_ref = private_file.as_posix()
    else:
        (media / "secret-link.txt").symlink_to(private_file)
        media_ref = "media/secret-link.txt"
    (collection / "Shared.md").write_text(
        "<!-- note_key: media-escape-key -->\n"
        "Q: shared\n"
        f"A: ![secret](<{media_ref}>)\n",
        encoding="utf-8",
    )
    _commit(collection, f"Add {escape_kind} media escape attempt")
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid media must fail before creating a GitHub repository"
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="media path.*outside.*media"):
        run_publish(SimpleNamespace(deck="Shared", repository="owner/media-escape"))

    assert private_file.read_text(encoding="utf-8") == "must never be published\n"
    assert not (collection / "collab" / "owner" / "media-escape").exists()


def test_publish_rejects_a_media_symlink_even_when_its_target_is_contained(
    tmp_path, monkeypatch
):
    collection = _setup_collection(tmp_path)
    DeckFileHarness().eject_default_note_types(collection / "note_types")
    media = collection / "media"
    media.mkdir()
    private_file = media / "unreferenced-private.txt"
    private_file.write_text("must never be published\n", encoding="utf-8")
    (media / "shared-link.txt").symlink_to(private_file)
    (collection / "Shared.md").write_text(
        "<!-- note_key: contained-symlink-key -->\n"
        "Q: shared\n"
        "A: ![secret](media/shared-link.txt)\n",
        encoding="utf-8",
    )
    _commit(collection, "Add contained media symlink")
    monkeypatch.setattr(
        "ankiops.collab.publish.GitHubHost.create_repo",
        lambda *_args, **_kwargs: pytest.fail(
            "symlinked media must fail before creating a GitHub repository"
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_root", lambda: collection
    )

    with pytest.raises(ValueError, match="symbolic link"):
        run_publish(
            SimpleNamespace(deck="Shared", repository="owner/contained-symlink")
        )

    assert private_file.read_text(encoding="utf-8") == "must never be published\n"
    assert not (collection / "collab" / "owner" / "contained-symlink").exists()


def test_update_checkpoints_local_edit_and_integrates_remote(collab_world, tmp_path):
    collection, source, remote = collab_world
    private_before = (collection / "Private.md").read_bytes()
    (source / "Local.md").write_text(
        "<!-- note_key: local-key -->\nQ: local\nA: edit\n", encoding="utf-8"
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert (source / "Local.md").exists()
    assert "remote answer" in (source / "Deck.md").read_text(encoding="utf-8")
    assert (
        _git(source, "rev-parse", "upstream/main").stdout.strip()
        == _git(remote, "rev-parse", "main").stdout.strip()
    )
    assert (collection / "Private.md").read_bytes() == private_before
    assert (
        "Save local deck changes for owner/repo"
        in _git(source, "log", "--format=%s").stdout
    )


def test_update_does_not_require_unrelated_local_draft_to_parse(
    collab_world, tmp_path, capsys
):
    _collection, source, remote = collab_world
    draft = source / "Draft.md"
    draft.write_text("work in progress\n", encoding="utf-8")
    upstream = tmp_path / "docs-repair"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Documentation Contributor")
    (upstream / "README.md").write_text("Upstream docs\n", encoding="utf-8")
    _commit(upstream, "Update docs")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert draft.read_text(encoding="utf-8") == "work in progress\n"
    assert (source / "README.md").read_text(encoding="utf-8") == "Upstream docs\n"
    assert "Apply to Anki" not in capsys.readouterr().out


def test_update_recommends_anki_only_for_applicable_files(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "readme-update"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Documentation Contributor")
    (upstream / "README.md").write_text("Documentation only\n", encoding="utf-8")
    _commit(upstream, "Document the deck")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "integrated upstream changes" in output
    assert "Apply to Anki" not in output

    _upstream_edit(tmp_path, remote, "upstream answer", "Anki-visible answer")
    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "ankiops fa" in output


def test_update_does_not_recommend_anki_for_unreferenced_assets(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "unreferenced-assets"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Auxiliary Asset Contributor")
    media = upstream / "media"
    media.mkdir()
    (media / "unused.txt").write_text("not referenced\n", encoding="utf-8")
    unreferenced_note_type_asset = (
        upstream / "note_types" / "AnkiOpsChoice" / "PrivateNotes.txt"
    )
    unreferenced_note_type_asset.write_text("not loaded\n", encoding="utf-8")
    _commit(upstream, "Add auxiliary assets")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "integrated upstream changes" in output
    assert "Apply to Anki" not in output


def test_update_recommends_anki_for_referenced_media_change(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "referenced-media"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Media Contributor")
    media = upstream / "media"
    media.mkdir()
    shared = media / "shared.txt"
    shared.write_text("version one\n", encoding="utf-8")
    deck = upstream / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "A: upstream answer", "A: [sound:shared.txt]"
        ),
        encoding="utf-8",
    )
    _commit(upstream, "Reference shared media")
    _git(upstream, "push", "origin", "main")
    run_update(SimpleNamespace(repository="owner/repo"))
    capsys.readouterr()

    shared.write_text("version two\n", encoding="utf-8")
    _commit(upstream, "Update shared media")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "Apply to Anki" in capsys.readouterr().out


def test_update_recommends_anki_for_referenced_note_type_asset(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "referenced-note-type"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Note Type Contributor")
    styling = upstream / "note_types" / "AnkiOpsStyling.css"
    styling.write_text(
        styling.read_text(encoding="utf-8") + "\n.card { opacity: 0.99; }\n",
        encoding="utf-8",
    )
    _commit(upstream, "Update shared note type styling")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "Apply to Anki" in capsys.readouterr().out


def test_update_recommends_anki_when_a_media_reference_is_removed(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "removed-media-reference"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Reference Removal Contributor")
    media = upstream / "media"
    media.mkdir()
    shared = media / "shared.txt"
    shared.write_text("shared\n", encoding="utf-8")
    deck = upstream / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "A: upstream answer", "A: [sound:shared.txt]"
        ),
        encoding="utf-8",
    )
    _commit(upstream, "Reference shared media")
    _git(upstream, "push", "origin", "main")
    run_update(SimpleNamespace(repository="owner/repo"))
    capsys.readouterr()

    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "A: [sound:shared.txt]", "A: answer without media"
        ),
        encoding="utf-8",
    )
    shared.unlink()
    _commit(upstream, "Remove shared media reference")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "Apply to Anki" in capsys.readouterr().out


def test_update_recommends_anki_for_renamed_unicode_deck(
    collab_world, tmp_path, capsys
):
    _collection, _source, remote = collab_world
    upstream = tmp_path / "unicode-deck-rename"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Unicode Contributor")
    _git(upstream, "mv", "Deck.md", "Déck Ω — spaced.md")
    _commit(upstream, "Rename deck with Unicode")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "ankiops fa" in capsys.readouterr().out


def test_conflict_preserves_versions_and_leaves_source_unchanged(
    collab_world, tmp_path, caplog
):
    caplog.set_level(logging.INFO)
    collection, source, remote = collab_world
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


def test_binary_conflict_stays_unresolved_until_its_placeholder_is_replaced(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    binary = source / "media" / "diagram.bin"
    binary.parent.mkdir()
    binary.write_bytes(b"base\x00diagram")
    base_commit = _commit(source, "Add shared binary")
    _git(source, "push", "upstream", "HEAD:main")
    _git(source, "update-ref", "refs/ankiops/integrated", base_commit)

    binary.write_bytes(b"local\x00diagram")
    upstream = tmp_path / "binary-upstream-edit"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Binary Contributor")
    (upstream / "media" / "diagram.bin").write_bytes(b"upstream\x00diagram")
    _commit(upstream, "Edit shared binary upstream")
    _git(upstream, "push", "origin", "main")
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_status = _git(source, "status", "--porcelain=v1").stdout

    with pytest.raises(ValueError, match="overlap"):
        run_update(SimpleNamespace(repository="owner/repo"))

    conflict_file = next(
        (collection / ".ankiops" / "conflicts").rglob("media/diagram.bin")
    )
    assert conflict_file.with_name("diagram.bin.base").read_bytes() == (
        b"base\x00diagram"
    )
    assert conflict_file.with_name("diagram.bin.local").read_bytes() == (
        b"local\x00diagram"
    )
    assert conflict_file.with_name("diagram.bin.upstream").read_bytes() == (
        b"upstream\x00diagram"
    )
    assert b"<<<<<<<" in conflict_file.read_bytes()

    with pytest.raises(ValueError, match="overlap"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert binary.read_bytes() == b"local\x00diagram"
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status

    conflict_file.write_bytes(
        conflict_file.with_name("diagram.bin.upstream").read_bytes()
    )
    run_update(SimpleNamespace(repository="owner/repo"))

    assert binary.read_bytes() == b"upstream\x00diagram"
    assert not list((collection / ".ankiops" / "conflicts").rglob("diagram.bin"))


def test_delete_modify_conflict_can_be_resolved_by_deleting_the_placeholder(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.unlink()
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    with pytest.raises(ValueError, match="overlap"):
        run_update(SimpleNamespace(repository="owner/repo"))

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    assert b"upstream answer" in conflict_file.with_name("Deck.md.base").read_bytes()
    assert not conflict_file.with_name("Deck.md.local").exists()
    assert (
        b"upstream correction"
        in conflict_file.with_name("Deck.md.upstream").read_bytes()
    )
    assert b"<<<<<<<" in conflict_file.read_bytes()

    conflict_file.unlink()
    run_update(SimpleNamespace(repository="owner/repo"))

    assert not deck.exists()
    assert not list((collection / ".ankiops" / "conflicts").rglob("Deck.md"))


def test_add_add_conflict_preserves_both_additions_and_accepts_an_edited_result(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    added = source / "Added.md"
    local_content = (
        b"<!-- note_key: added-key -->\nQ: locally added question\nA: local answer\n"
    )
    upstream_content = (
        b"<!-- note_key: added-key -->\n"
        b"Q: upstream added question\nA: upstream answer\n"
    )
    added.write_bytes(local_content)
    _commit(source, "Add local deck")
    upstream = tmp_path / "add-add-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Add/Add Contributor")
    (upstream / "Added.md").write_bytes(upstream_content)
    _commit(upstream, "Add deck upstream")
    _git(upstream, "push", "origin", "main")

    with pytest.raises(ValueError, match="Added.md"):
        run_update(SimpleNamespace(repository="owner/repo"))

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Added.md"))
    assert not conflict_file.with_name("Added.md.base").exists()
    assert conflict_file.with_name("Added.md.local").read_bytes() == local_content
    assert conflict_file.with_name("Added.md.upstream").read_bytes() == upstream_content
    assert b"<<<<<<<" in conflict_file.read_bytes()

    conflict_file.write_text(
        "<!-- note_key: added-key -->\nQ: jointly added question\nA: combined answer\n",
        encoding="utf-8",
    )
    run_update(SimpleNamespace(repository="owner/repo"))

    assert b"combined answer" in added.read_bytes()
    assert not list((collection / ".ankiops" / "conflicts").rglob("Added.md"))


def test_rename_rename_conflict_can_keep_one_name_and_delete_the_other_paths(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    _git(source, "mv", "Deck.md", "Local name.md")
    upstream = tmp_path / "rename-rename-upstream"
    _git(tmp_path, "clone", str(remote), str(upstream))
    _configure(upstream, "Rename Contributor")
    _git(upstream, "mv", "Deck.md", "Upstream name.md")
    _commit(upstream, "Rename deck upstream")
    _git(upstream, "push", "origin", "main")

    with pytest.raises(ValueError, match="overlap"):
        run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        conflict_root = Path(str(operation["recovery_ref"]))
    finally:
        db.close()
    original = conflict_root / "Deck.md"
    local = conflict_root / "Local name.md"
    upstream_name = conflict_root / "Upstream name.md"
    assert original.with_name("Deck.md.base").exists()
    assert local.with_name("Local name.md.local").exists()
    assert upstream_name.with_name("Upstream name.md.upstream").exists()
    assert all(
        b"<<<<<<<" in path.read_bytes() for path in (original, local, upstream_name)
    )

    original.unlink()
    local.write_bytes(local.with_name("Local name.md.local").read_bytes())
    upstream_name.unlink()
    run_update(SimpleNamespace(repository="owner/repo"))

    assert (source / "Local name.md").exists()
    assert not (source / "Deck.md").exists()
    assert not (source / "Upstream name.md").exists()


def test_conflict_preserves_versions_for_unicode_deck_path(collab_world, tmp_path):
    collection, source, remote = collab_world
    deck_name = "Déck Ω — punctuation!.md"
    _git(source, "mv", "Deck.md", deck_name)
    _commit(source, "Rename collab deck with Unicode")
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
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        conflict_root = Path(str(operation["recovery_ref"]))
    finally:
        db.close()
    assert (conflict_root / deck_name).read_text(encoding="utf-8")
    for suffix in ("base", "local", "upstream"):
        assert (conflict_root / f"{deck_name}.{suffix}").read_text(encoding="utf-8")


def test_conflict_can_be_resolved_by_editing_preserved_markdown_and_retrying_update(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
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
        "<!-- note_key: collab-key -->\nQ: collab question\nA: combined answer\n",
        encoding="utf-8",
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    assert not GitRepository(source).unmerged_paths()
    assert "combined answer" in deck.read_text(encoding="utf-8")
    assert not list((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_update_does_not_apply_a_resolution_after_upstream_advances(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    _upstream_edit(tmp_path, remote, "upstream answer", "first upstream answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        first_operation = db.get_collab_operation("collab/owner/repo")
        assert first_operation is not None
        first_recovery = Path(str(first_operation["recovery_ref"]))
    finally:
        db.close()
    stale_resolution = first_recovery / "Deck.md"
    stale_resolution.write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: stale resolution\n",
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "first upstream answer", "newer upstream answer")

    with pytest.raises(ValueError, match="upstream advanced"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert "local answer" in deck.read_text(encoding="utf-8")
    assert "stale resolution" not in deck.read_text(encoding="utf-8")
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert "stale resolution" in stale_resolution.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        refreshed = db.get_collab_operation("collab/owner/repo")
        assert refreshed is not None
        refreshed_recovery = Path(str(refreshed["recovery_ref"]))
    finally:
        db.close()
    assert refreshed_recovery != first_recovery
    assert "newer upstream answer" in (
        refreshed_recovery / "Deck.md.upstream"
    ).read_text(encoding="utf-8")


def test_update_requires_confirmation_when_advanced_upstream_makes_merge_clean(
    collab_world, tmp_path
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "local answer"),
        encoding="utf-8",
    )
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    before_status = _git(source, "status", "--porcelain=v1").stdout
    _upstream_edit(tmp_path, remote, "upstream answer", "first upstream answer")
    with pytest.raises(ValueError):
        run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        first_operation = db.get_collab_operation("collab/owner/repo")
        assert first_operation is not None
        first_recovery = Path(str(first_operation["recovery_ref"]))
    finally:
        db.close()
    prior_resolution = first_recovery / "Deck.md"
    prior_resolution.write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: carefully combined\n",
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "first upstream answer", "local answer")

    with pytest.raises(ValueError, match="merge is now clean"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    assert "carefully combined" in prior_resolution.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        refreshed = db.get_collab_operation("collab/owner/repo")
        assert refreshed is not None
        refreshed_recovery = Path(str(refreshed["recovery_ref"]))
    finally:
        db.close()
    confirmation = refreshed_recovery / "CONFIRM_CLEAN_MERGE"
    assert refreshed_recovery != first_recovery
    assert confirmation.exists()
    assert (
        b"carefully combined" in (refreshed_recovery / "Deck.md.previous").read_bytes()
    )
    assert b"local answer" in (refreshed_recovery / "Deck.md.candidate").read_bytes()
    assert b"local answer" in (refreshed_recovery / "Deck.md.upstream").read_bytes()

    with pytest.raises(ValueError, match="confirmation is still required"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(source, "status", "--porcelain=v1").stdout == before_status
    confirmation.unlink()
    run_update(SimpleNamespace(repository="owner/repo"))

    assert "local answer" in deck.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_submit_conflict_is_resumed_with_update(collab_world, tmp_path):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "local contribution"
        ),
        encoding="utf-8",
    )
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    with pytest.raises(ValueError, match="collab update owner/repo"):
        run_submit(SimpleNamespace(repository="owner/repo", title=None))

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: combined\n",
        encoding="utf-8",
    )
    run_update(SimpleNamespace(repository="owner/repo"))

    assert "combined" in deck.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_resolving_a_revision_conflict_keeps_the_existing_pull_request(
    collab_world, tmp_path, monkeypatch
):
    collection, source, upstream_remote = collab_world
    publish = tmp_path / "revision-conflict-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return f"https://github.com/owner/repo/pull/{len(created_prs)}"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.update_pr", lambda *_args, **_kwargs: None
    )

    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first submitted answer"
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(repository="owner/repo", title="Clarify answer")
    run_submit(args)

    db = SyncState.open(collection)
    try:
        submitted = db.get_collab_operation("collab/owner/repo")
        assert submitted is not None
        original_branch = submitted["publish_branch"]
        original_sha = submitted["pushed_sha"]
        original_url = submitted["pr_url"]
    finally:
        db.close()

    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first submitted answer", "revised local answer"
        ),
        encoding="utf-8",
    )
    _upstream_edit(
        tmp_path, upstream_remote, "upstream answer", "overlapping upstream answer"
    )
    with pytest.raises(ValueError, match="overlap"):
        run_submit(args)

    conflict_file = next((collection / ".ankiops" / "conflicts").rglob("Deck.md"))
    conflict_file.write_text(
        "<!-- note_key: collab-key -->\nQ: collab question\nA: combined answer\n",
        encoding="utf-8",
    )
    run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        resumed = db.get_collab_operation("collab/owner/repo")
        assert resumed is not None
        assert resumed["state"] == "pr_open"
        assert resumed["publish_branch"] == original_branch
        assert resumed["pushed_sha"] == original_sha
        assert resumed["pr_url"] == original_url
    finally:
        db.close()

    run_submit(args)

    assert len(created_prs) == 1
    db = SyncState.open(collection)
    try:
        updated = db.get_collab_operation("collab/owner/repo")
        assert updated is not None
        assert updated["publish_branch"] == original_branch
        assert updated["pr_url"] == original_url
        assert updated["pushed_sha"] != original_sha
    finally:
        db.close()


def test_failed_update_fetch_leaves_repository_exactly_unchanged(
    collab_world, monkeypatch
):
    collection, source, _remote = collab_world
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
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "integrating"
    finally:
        db.close()


def test_update_refuses_to_overwrite_a_concurrent_source_edit(
    collab_world, tmp_path, monkeypatch
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    before_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    integrated_before = _git(
        source, "rev-parse", "refs/ankiops/integrated"
    ).stdout.strip()
    _upstream_edit(tmp_path, remote, "upstream answer", "remote answer")

    original_save = SyncState.save_collab_operation
    injected = False

    def inject_edit_before_apply(
        state, source_path, operation_id, kind, operation_state, **values
    ):
        nonlocal injected
        original_save(
            state,
            source_path,
            operation_id,
            kind,
            operation_state,
            **values,
        )
        if operation_state == "applying" and not injected:
            injected = True
            deck.write_text(
                deck.read_text(encoding="utf-8").replace(
                    "upstream answer", "concurrent answer"
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(SyncState, "save_collab_operation", inject_edit_before_apply)

    with pytest.raises(ValueError, match="changed while its update was prepared"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert "concurrent answer" in deck.read_text(encoding="utf-8")
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == before_head
    assert (
        _git(source, "rev-parse", "refs/ankiops/integrated").stdout.strip()
        == integrated_before
    )


def test_submit_commits_only_source_and_reuses_operation(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, _remote = collab_world
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
    _git(source, "add", "Deck.md")
    (source / "Deck.md").write_text(
        (source / "Deck.md")
        .read_text(encoding="utf-8")
        .replace("submitted answer", "final unstaged answer"),
        encoding="utf-8",
    )
    (source / "draft-notes.txt").write_text("untracked draft\n", encoding="utf-8")
    (source / ".gitignore").write_text("scratch.tmp\n", encoding="utf-8")
    (source / "scratch.tmp").write_text("must stay private\n", encoding="utf-8")
    (source / "private-reference.link").symlink_to("../../../Private.md")

    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    created_pr = []

    def create_pr(*_args, **kwargs):
        created_pr.append(kwargs)
        return "https://github.test/owner/repo/pull/1"

    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        create_pr,
    )

    args = SimpleNamespace(repository="owner/repo", title=None)
    run_submit(args)
    first_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    first_refs = _git(publish, "for-each-ref", "--format=%(refname)").stdout
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == first_head
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == first_refs
    assert created_pr[0]["head"].startswith("contributor:ankiops/")
    assert created_pr[0]["title"] == "Update Deck"
    uploaded = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    uploaded_deck = _git(publish, "show", f"{uploaded}:Deck.md").stdout
    assert "final unstaged answer" in uploaded_deck
    assert _git(publish, "show", f"{uploaded}:draft-notes.txt").stdout == (
        "untracked draft\n"
    )
    assert _git(
        publish, "ls-tree", uploaded, "private-reference.link"
    ).stdout.startswith("120000 blob ")
    assert (
        _git(publish, "show", f"{uploaded}:private-reference.link").stdout
        == "../../../Private.md"
    )
    assert _git(publish, "show", f"{uploaded}:scratch.tmp", check=False).returncode != 0
    assert _git(source, "log", "-1", "--format=%s").stdout.strip() == (
        "Save local deck changes for owner/repo"
    )
    assert private.read_text(encoding="utf-8") == "private dirty\n"
    assert _git(collection, "status", "--short").stdout.strip() == "M Private.md"
    output = capsys.readouterr().out
    assert "Pull request: https://github.test/owner/repo/pull/1" in output
    assert "Next" not in output
    assert "Review" not in output
    assert "Anki" not in output
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/1"
    finally:
        db.close()


def test_submit_derives_human_title_for_unicode_deck_rename(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "unicode-title-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _git(source, "mv", "Deck.md", "Déck Ω — spaced.md")
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []
    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return "https://github.test/owner/repo/pull/unicode-title"

    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        create_pr,
    )

    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    assert created_prs[0]["title"] == "Rename Deck to Déck Ω — spaced"


def test_submit_updates_open_pull_request_with_later_draft(
    collab_world, tmp_path, monkeypatch, capsys
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first submitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    publish_target_calls = []

    def publish_target(*_args):
        publish_target_calls.append(True)
        if len(publish_target_calls) > 1:
            pytest.fail("an existing PR revision must retain its publish remote")
        return "contributor/repo", "contributor"

    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target", publish_target
    )
    created_prs = []

    def create_pr(*_args, **_kwargs):
        created_prs.append(_kwargs)
        return "https://github.test/owner/repo/pull/1"

    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        create_pr,
    )
    args = SimpleNamespace(repository="owner/repo", title="First contribution")

    run_submit(args)
    capsys.readouterr()
    first_uploaded = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first submitted answer", "review feedback answer"
        ),
        encoding="utf-8",
    )

    run_submit(args)
    output = capsys.readouterr().out

    second_uploaded = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    upstream_head = _git(source, "rev-parse", "upstream/main").stdout.strip()
    assert second_uploaded != first_uploaded
    assert (
        _git(
            publish,
            "rev-list",
            "--count",
            f"{upstream_head}..{second_uploaded}",
        ).stdout.strip()
        == "1"
    )
    assert _git(source, "status", "--short").stdout == ""
    assert len(created_prs) == 1
    assert len(publish_target_calls) == 1
    assert "pull request updated" in output


def test_submit_refuses_to_retarget_an_open_pr_from_a_different_github_account(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "account-bound-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first account-bound answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/31",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="First revision"))
    uploaded_before = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()

    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first account-bound answer", "second account-bound answer"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN", head_owner="contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.login", lambda *_args: "owner"
    )
    permission_checks = []

    def no_push_access(_host, slug):
        permission_checks.append(slug)
        return False

    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.can_push", no_push_access)

    with pytest.raises(ValueError) as error:
        run_submit(SimpleNamespace(repository="owner/repo", title="Second revision"))

    assert str(error.value) == (
        "GitHub CLI is authenticated as @owner, but this pull request belongs to "
        "@contributor. Nothing was sent. Switch accounts, then retry: gh auth switch "
        "--hostname github.com --user contributor"
    )
    assert permission_checks == ["contributor/repo"]
    assert "second account-bound answer" in deck.read_text(encoding="utf-8")
    assert (
        _git(
            publish,
            "for-each-ref",
            "--format=%(objectname)",
            "refs/heads/ankiops/*",
        ).stdout.strip()
        == uploaded_before
    )


def test_submit_refuses_wrong_account_title_only_revision(collab_world, monkeypatch):
    collection, source, _remote = collab_world
    prepared_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "account-bound-title",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/account-bound-title",
            pr_url="https://github.com/owner/repo/pull/33",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN", head_owner="contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.login", lambda *_args: "owner"
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.can_push",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.update_pr",
        lambda *_args, **_kwargs: pytest.fail(
            "wrong-account title update must not reach GitHub"
        ),
    )

    with pytest.raises(ValueError) as error:
        run_submit(SimpleNamespace(repository="owner/repo", title="Retitle PR"))

    assert str(error.value) == (
        "GitHub CLI is authenticated as @owner, but this pull request belongs to "
        "@contributor. Nothing was sent. Switch accounts, then retry: gh auth switch "
        "--hostname github.com --user contributor"
    )


def test_submit_permission_lookup_failure_is_distinct_and_retry_safe(
    collab_world, monkeypatch
):
    collection, source, _remote = collab_world
    prepared_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "permission-lookup",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/permission-lookup",
            pr_url="https://github.com/owner/repo/pull/34",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN", head_owner="contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.login", lambda *_args: "owner"
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.can_push",
        lambda *_args: (_ for _ in ()).throw(ValueError("permission API unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.repo_info",
        lambda *_args: pytest.fail("permission checks must preserve API failures"),
    )

    with pytest.raises(ValueError) as error:
        run_submit(SimpleNamespace(repository="owner/repo", title="Retitle PR"))

    message = str(error.value)
    assert "Could not confirm GitHub push access to contributor/repo" in message
    assert "Nothing was sent" in message
    assert "Retrying is safe: ankiops collab submit owner/repo" in message
    assert "permission API unavailable" in message
    assert "gh auth switch" not in message
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.com/owner/repo/pull/34"
    finally:
        db.close()


def test_submit_allows_collaborator_with_push_access_to_revise_owner_branch(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "owner-direct-write.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first direct-write answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("owner/repo", "owner"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/32",
    )
    args = SimpleNamespace(repository="owner/repo", title=None)

    run_submit(args)
    uploaded_before = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first direct-write answer", "collaborator revision answer"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN", head_owner="owner"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.login", lambda *_args: "collaborator"
    )
    permission_checks = []

    def repo_info(_host, slug):
        permission_checks.append(slug)
        return True

    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.can_push", repo_info)

    run_submit(args)

    uploaded_after = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    assert uploaded_after != uploaded_before
    assert (
        "collaborator revision answer"
        in _git(publish, "show", f"{uploaded_after}:Deck.md").stdout
    )
    assert permission_checks == ["owner/repo"]


def test_submit_uses_live_merged_pull_request_state(collab_world, monkeypatch):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "already uploaded answer"
        ),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Uploaded contribution")
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "live-state",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/live-state",
            pr_url="https://github.com/owner/repo/pull/7",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="MERGED"),
    )

    with pytest.raises(ValueError, match="has been merged.*collab update"):
        run_submit(SimpleNamespace(repository="owner/repo", title=None))


def test_submit_replaces_a_pull_request_closed_without_merge(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
    publish = tmp_path / "closed-pr-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "resubmitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return f"https://github.com/owner/repo/pull/{len(created_prs)}"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="CLOSED"),
    )
    args = SimpleNamespace(repository="owner/repo", title=None)

    run_submit(args)
    first_branch = created_prs[0]["head"].split(":", 1)[1]
    run_submit(args)

    assert len(created_prs) == 2
    second_branch = created_prs[1]["head"].split(":", 1)[1]
    assert second_branch != first_branch
    assert (
        _git(
            publish, "show-ref", "--verify", f"refs/heads/{first_branch}", check=False
        ).returncode
        != 0
    )
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["pr_url"] == "https://github.com/owner/repo/pull/2"
    finally:
        db.close()


def test_submit_closed_pr_never_deletes_an_advanced_or_reused_head_branch(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
    publish = tmp_path / "closed-advanced-head.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "resubmitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return f"https://github.com/owner/repo/pull/{len(created_prs)}"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    args = SimpleNamespace(repository="owner/repo", title=None)
    run_submit(args)
    first_branch = created_prs[0]["head"].split(":", 1)[1]

    reused = tmp_path / "closed-branch-reused"
    _git(tmp_path, "clone", str(publish), str(reused))
    _configure(reused, "Other User")
    _git(reused, "checkout", first_branch)
    (reused / "Reused.md").write_text("must survive cleanup\n", encoding="utf-8")
    reused_head = _commit(reused, "Reuse closed PR branch")
    _git(reused, "push", "origin", first_branch)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="CLOSED",
            head_owner="contributor",
            head_branch=first_branch,
            head_sha=reused_head,
        ),
    )

    run_submit(args)

    assert _git(publish, "rev-parse", first_branch).stdout.strip() == reused_head
    assert _git(publish, "show", f"{first_branch}:Reused.md").stdout == (
        "must survive cleanup\n"
    )
    assert len(created_prs) == 2
    second_branch = created_prs[1]["head"].split(":", 1)[1]
    assert second_branch != first_branch
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["publish_branch"] == second_branch
    finally:
        db.close()


def test_status_uses_live_closed_pull_request_state(collab_world, monkeypatch, capsys):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "closed contribution"
        ),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Closed contribution")
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "closed-state",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/closed-state",
            pr_url="https://github.com/owner/repo/pull/8",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="CLOSED"),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Submission: pull request closed without merge" in output
    assert "Submit again: ankiops collab submit owner/repo" in output
    assert "Review" not in output


def test_status_open_pull_request_has_no_submitter_action(collab_world, capsys):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "open contribution"
        ),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Open contribution")
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "open-state",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/open-state",
            pr_url="https://github.test/owner/repo/pull/9",
        )
    finally:
        db.close()

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Local contribution: 1 change uploaded" in output
    assert "ready to submit" not in output
    assert "Submission: pull request open" in output
    assert "Next" not in output
    assert "Submit contribution" not in output
    assert "Review" not in output


def test_open_pull_request_is_not_accepted_when_its_content_appears_upstream(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, upstream_remote = collab_world
    publish = tmp_path / "still-open-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/12",
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN"),
    )

    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "independently repeated answer"
        ),
        encoding="utf-8",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Clarify answer"))
    db = SyncState.open(collection)
    try:
        submitted = db.get_collab_operation("collab/owner/repo")
        assert submitted is not None
        contribution_branch = str(submitted["publish_branch"])
    finally:
        db.close()

    upstream = tmp_path / "independent-upstream-change"
    _git(tmp_path, "clone", str(upstream_remote), str(upstream))
    _configure(upstream, "Independent Maintainer")
    (upstream / "Deck.md").write_bytes(deck.read_bytes())
    _commit(upstream, "Make the same change independently")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        remaining = db.get_collab_operation("collab/owner/repo")
        assert remaining is not None
        assert remaining["state"] == "pr_open"
        assert remaining["pr_url"] == "https://github.com/owner/repo/pull/12"
    finally:
        db.close()
    assert (
        _git(
            publish,
            "show-ref",
            "--verify",
            f"refs/heads/{contribution_branch}",
            check=False,
        ).returncode
        == 0
    )

    capsys.readouterr()
    run_status(SimpleNamespace(repository="owner/repo"))
    output = capsys.readouterr().out
    assert "Submission: pull request open" in output
    assert "Submission: merged" not in output


def test_update_pr_lookup_failure_retains_submission_and_remote_branch(
    collab_world, tmp_path, monkeypatch
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "independently repeated answer"
        ),
        encoding="utf-8",
    )
    upstream_tree = _git(source, "rev-parse", "upstream/main^{tree}").stdout.strip()
    prepared_head = _commit(source, "Uploaded contribution")
    _git(source, "update-ref", "refs/ankiops/uploaded", prepared_head)

    publish = tmp_path / "lookup-failure-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _git(source, "remote", "add", "publish", str(publish))
    branch = "ankiops/lookup-failure"
    _git(source, "push", "publish", f"HEAD:refs/heads/{branch}")
    _upstream_edit(
        tmp_path,
        remote,
        "upstream answer",
        "independently repeated answer",
    )
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "lookup-failure",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=upstream_tree,
            publish_branch=branch,
            pr_url="https://github.com/owner/repo/pull/13",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: (_ for _ in ()).throw(ValueError("API unavailable")),
    )

    run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.com/owner/repo/pull/13"
    finally:
        db.close()
    assert (
        _git(source, "rev-parse", "refs/ankiops/uploaded").stdout.strip()
        == prepared_head
    )
    assert (
        _git(
            publish,
            "show-ref",
            "--verify",
            f"refs/heads/{branch}",
            check=False,
        ).returncode
        == 0
    )


def test_status_pr_lookup_failure_never_reports_tree_fallback_as_merged(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "independently repeated answer"
        ),
        encoding="utf-8",
    )
    upstream_tree = _git(source, "rev-parse", "upstream/main^{tree}").stdout.strip()
    prepared_head = _commit(source, "Uploaded contribution")
    _upstream_edit(
        tmp_path,
        remote,
        "upstream answer",
        "independently repeated answer",
    )
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "lookup-failure-status",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=upstream_tree,
            publish_branch="ankiops/lookup-failure-status",
            pr_url="https://github.com/owner/repo/pull/14",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: (_ for _ in ()).throw(ValueError("API unavailable")),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Submission: pull request state unavailable" in output
    assert "Submission: merged" not in output
    assert "Retry status: ankiops collab status owner/repo" in output


def test_open_pull_request_unicode_change_survives_unrelated_upstream_update(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, upstream_remote = collab_world
    unicode_name = "Déck Ω — spaced.md"
    _git(source, "mv", "Deck.md", unicode_name)
    prepared_head = _commit(source, "Uploaded Unicode rename")
    _git(source, "update-ref", "refs/ankiops/uploaded", prepared_head)
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "unicode-open",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/unicode-open",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/13",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN"),
    )

    upstream = tmp_path / "unrelated-unicode-upstream"
    _git(tmp_path, "clone", str(upstream_remote), str(upstream))
    _configure(upstream, "Documentation Maintainer")
    (upstream / "README.md").write_text("Unrelated docs\n", encoding="utf-8")
    _commit(upstream, "Add unrelated documentation")
    _git(upstream, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Update pull request: ankiops collab submit owner/repo" in output
    assert "Apply to Anki" not in output
    assert (source / unicode_name).exists()
    assert (source / "README.md").read_text(encoding="utf-8") == "Unrelated docs\n"
    db = SyncState.open(collection)
    try:
        remaining = db.get_collab_operation("collab/owner/repo")
        assert remaining is not None
        assert remaining["state"] == "pr_open"
        assert remaining["pr_url"] == "https://github.com/owner/repo/pull/13"
    finally:
        db.close()
    assert _git(source, "rev-parse", "refs/ankiops/uploaded").stdout.strip()


def test_update_preserves_an_open_pr_edited_on_github_and_never_recommends_upload(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, upstream_remote = collab_world
    publish = tmp_path / "github-edited-open-update.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "submitted wording"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/28",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Submitted wording"))
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        branch = str(operation["publish_branch"])
    finally:
        db.close()

    edited_head = tmp_path / "github-edited-update-head"
    _git(tmp_path, "clone", str(publish), str(edited_head))
    _configure(edited_head, "PR Maintainer")
    _git(edited_head, "checkout", branch)
    edited_deck = edited_head / "Deck.md"
    edited_deck.write_text(
        edited_deck.read_text(encoding="utf-8").replace(
            "submitted wording", "maintainer review wording"
        ),
        encoding="utf-8",
    )
    maintainer_head = _commit(edited_head, "Edit open PR")
    _git(edited_head, "push", "origin", branch)

    upstream = tmp_path / "unrelated-update-after-pr-edit"
    _git(tmp_path, "clone", str(upstream_remote), str(upstream))
    _configure(upstream, "Upstream Maintainer")
    (upstream / "README.md").write_text("Unrelated docs\n", encoding="utf-8")
    _commit(upstream, "Update documentation")
    _git(upstream, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="OPEN",
            head_owner="contributor",
            head_branch=branch,
            head_sha=maintainer_head,
        ),
    )

    capsys.readouterr()
    run_update(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Inspect GitHub changes before another upload" in output
    assert "Update pull request: ankiops collab submit owner/repo" not in output
    assert (source / "README.md").read_text(encoding="utf-8") == "Unrelated docs\n"
    assert "submitted wording" in deck.read_text(encoding="utf-8")
    assert _git(publish, "rev-parse", branch).stdout.strip() == maintainer_head
    db = SyncState.open(collection)
    try:
        remaining = db.get_collab_operation("collab/owner/repo")
        assert remaining is not None
        assert remaining["state"] == "pr_open"
    finally:
        db.close()


def test_submit_retry_after_failed_integration(collab_world, tmp_path, monkeypatch):
    _collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/integration",
    )
    args = SimpleNamespace(repository="owner/repo", title="Retry integration")

    with pytest.raises(ValueError, match="Retrying is safe"):
        run_submit(args)
    run_submit(args)

    assert fetches == 2
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout.strip()


def test_submit_retry_after_interruption_while_applying(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/apply",
    )
    args = SimpleNamespace(repository="owner/repo", title="Retry apply")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "applying"
    finally:
        db.close()

    run_submit(args)

    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout.strip()


def test_update_retry_after_apply_repairs_integrated_and_upstream_refs(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, upstream_remote = collab_world
    integrated_before = _git(
        source, "rev-parse", "refs/ankiops/integrated"
    ).stdout.strip()
    upstream_head = _upstream_edit(
        tmp_path, upstream_remote, "upstream answer", "new upstream answer"
    )
    original_reset_hard = GitRepository.reset_hard
    interrupted = False

    def interrupt_after_source_reset(source_git, ref):
        nonlocal interrupted
        original_reset_hard(source_git, ref)
        if source_git.root == source and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(GitRepository, "reset_hard", interrupt_after_source_reset)

    with pytest.raises(KeyboardInterrupt):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert (
        _git(source, "rev-parse", "refs/ankiops/integrated").stdout.strip()
        == integrated_before
    )
    assert _git(source, "rev-parse", "upstream/main").stdout.strip() != upstream_head

    run_update(SimpleNamespace(repository="owner/repo"))

    assert (
        _git(source, "rev-parse", "refs/ankiops/integrated").stdout.strip()
        == upstream_head
    )
    assert _git(source, "rev-parse", "upstream/main").stdout.strip() == upstream_head


def test_submit_retry_after_failed_pr_reuses_pushed_branch(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", fail_once)
    args = SimpleNamespace(repository="owner/repo", title="Retry PR")

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
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/2"
    finally:
        db.close()


def test_submit_retry_after_failed_push_reuses_commit_and_branch(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/3",
    )
    args = SimpleNamespace(repository="owner/repo", title="Retry push")

    with pytest.raises(ValueError, match="did not reach GitHub"):
        run_submit(args)
    failed_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    db = SyncState.open(collection)
    try:
        failed_operation = db.get_collab_operation("collab/owner/repo")
        assert failed_operation is not None
        branch = failed_operation["publish_branch"]
        assert failed_operation["state"] == "push_failed"
    finally:
        db.close()
    run_submit(args)

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == failed_head
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["publish_branch"] == branch
        assert operation["state"] == "pr_open"
    finally:
        db.close()


@pytest.mark.parametrize("failure_state", ["ready", "push_failed"])
def test_submit_retry_after_status_fetch_keeps_the_integrated_parent(
    collab_world, tmp_path, monkeypatch, failure_state
):
    collection, source, upstream = collab_world
    publish = tmp_path / "push-retry-after-status.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally submitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote
    original_push = GitRepository.push
    pushes = 0
    target_lookups = 0
    created_prs = 0

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    def fail_once(source_git, remote_name, source_ref, branch):
        nonlocal pushes
        pushes += 1
        if failure_state == "push_failed" and pushes == 1:
            raise subprocess.CalledProcessError(1, ["git", "push"])
        original_push(source_git, remote_name, source_ref, branch)

    def publish_target(*_args):
        nonlocal target_lookups
        target_lookups += 1
        if failure_state == "ready" and target_lookups == 1:
            raise ValueError("simulated GitHub setup failure")
        return "contributor/repo", "contributor"

    def create_pr(*_args, **_kwargs):
        nonlocal created_prs
        created_prs += 1
        return "https://github.test/owner/repo/pull/31"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(GitRepository, "push", fail_once)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        publish_target,
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        create_pr,
    )
    args = SimpleNamespace(repository="owner/repo", title="Safe retry")

    expected_error = (
        "did not reach GitHub"
        if failure_state == "push_failed"
        else "GitHub setup did not finish"
    )
    with pytest.raises(ValueError, match=expected_error):
        run_submit(args)
    db = SyncState.open(collection)
    try:
        failed_operation = db.get_collab_operation("collab/owner/repo")
        assert failed_operation is not None
        assert failed_operation["state"] == failure_state
    finally:
        db.close()
    integrated_parent = _git(
        source, "rev-parse", "refs/ankiops/integrated"
    ).stdout.strip()

    advanced_upstream = _upstream_edit(
        tmp_path, upstream, "upstream answer", "later upstream answer"
    )
    run_status(SimpleNamespace(repository="owner/repo"))
    assert _git(source, "rev-parse", "upstream/main").stdout.strip() == (
        advanced_upstream
    )

    run_submit(args)

    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        submitted_commit = str(operation["pushed_sha"])
    finally:
        db.close()
    assert _git(source, "rev-parse", f"{submitted_commit}^").stdout.strip() == (
        integrated_parent
    )
    assert integrated_parent != advanced_upstream
    assert created_prs == 1
    assert _git(
        publish, "for-each-ref", "--format=%(refname)", "refs/heads/ankiops/*"
    ).stdout.splitlines() == [f"refs/heads/{operation['publish_branch']}"]


def test_submit_retry_after_interruption_after_push(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/pushed",
    )
    args = SimpleNamespace(repository="owner/repo", title="Retry pushed commit")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    run_submit(args)

    assert pushes == 1


def test_submit_retry_after_interruption_after_pr_creation(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr", interrupt_after_pr
    )
    args = SimpleNamespace(repository="owner/repo", title="Retry created PR")

    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    run_submit(args)

    assert pr_calls == 2
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == "https://github.test/owner/repo/pull/existing"
    finally:
        db.close()


def test_update_after_squash_merge_cleans_submission_state(
    collab_world, tmp_path, monkeypatch
):
    collection, source, upstream_remote = collab_world
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
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/4",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Merged change"))

    merged = tmp_path / "squash-merge"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged, "Maintainer")
    (merged / "Deck.md").write_bytes(deck.read_bytes())
    _commit(merged, "Squash merged contribution")
    _git(merged, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()
    assert _git(publish, "for-each-ref", "--format=%(refname)").stdout == ""


def test_update_uses_live_merge_as_the_uploaded_content_base(
    collab_world, tmp_path, monkeypatch, capsys
):
    collection, source, upstream_remote = collab_world
    publish = tmp_path / "maintainer-edited-merge.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "uploaded wording"),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/9",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Uploaded wording"))
    submitted_branch = _git(
        publish, "for-each-ref", "--format=%(refname:short)", "refs/heads/ankiops/*"
    ).stdout.strip()

    edited_head = tmp_path / "maintainer-edited-pr-head"
    _git(tmp_path, "clone", str(publish), str(edited_head))
    _configure(edited_head, "PR Maintainer")
    _git(edited_head, "checkout", submitted_branch)
    edited_head_deck = edited_head / "Deck.md"
    edited_head_deck.write_text(
        edited_head_deck.read_text(encoding="utf-8").replace(
            "uploaded wording", "maintainer-adjusted merged wording"
        ),
        encoding="utf-8",
    )
    maintainer_head = _commit(edited_head, "Maintainer edits PR head")
    _git(edited_head, "push", "origin", submitted_branch)

    merged = tmp_path / "maintainer-edited-upstream"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged, "Maintainer")
    _git(merged, "remote", "add", "publish", str(publish))
    _git(merged, "fetch", "publish", submitted_branch)
    _git(
        merged,
        "merge",
        "--no-ff",
        f"publish/{submitted_branch}",
        "-m",
        "Merge contribution with maintainer edit",
    )
    _git(merged, "push", "origin", "main")
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="MERGED",
            head_owner="contributor",
            head_branch=submitted_branch,
            head_sha=maintainer_head,
        ),
    )

    capsys.readouterr()
    run_update(SimpleNamespace(repository="owner/repo"))

    assert "maintainer-adjusted merged wording" in deck.read_text(encoding="utf-8")
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()
    assert (
        _git(
            publish,
            "show-ref",
            "--verify",
            f"refs/heads/{submitted_branch}",
            check=False,
        ).returncode
        != 0
    )
    assert "Submission branch: kept" not in capsys.readouterr().out


def test_update_cleanup_failure_leaves_local_deck_untouched(
    collab_world, tmp_path, monkeypatch
):
    collection, source, remote = collab_world
    publish = tmp_path / "cleanup-failure.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _git(source, "remote", "add", "publish", str(publish))
    _git(source, "push", "publish", "HEAD:refs/heads/ankiops/cleanup")
    prepared_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    integrated_before = _git(
        source, "rev-parse", "refs/ankiops/integrated"
    ).stdout.strip()
    _upstream_edit(tmp_path, remote, "upstream answer", "merged upstream answer")
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "cleanup-failure",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/cleanup",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/10",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="MERGED"),
    )
    monkeypatch.setattr(
        GitRepository,
        "delete_remote_branch_with_lease",
        lambda *_args: None,
    )

    with pytest.raises(ValueError, match="could not be removed"):
        run_update(SimpleNamespace(repository="owner/repo"))

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == prepared_head
    assert (
        _git(source, "rev-parse", "refs/ankiops/integrated").stdout.strip()
        == integrated_before
    )
    assert "upstream answer" in (source / "Deck.md").read_text(encoding="utf-8")


def test_update_after_squash_merge_preserves_edits_made_after_upload(
    collab_world, tmp_path, monkeypatch
):
    collection, source, upstream_remote = collab_world
    publish = tmp_path / "merged-with-later-edit.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "uploaded answer"),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.test/owner/repo/pull/5",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="Uploaded answer"))
    uploaded_content = deck.read_bytes()

    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "uploaded answer", "later review answer"
        ),
        encoding="utf-8",
    )
    merged = tmp_path / "squash-merge-with-later-edit"
    _git(tmp_path, "clone", str(upstream_remote), str(merged))
    _configure(merged, "Maintainer")
    (merged / "Deck.md").write_bytes(uploaded_content)
    (merged / "README.md").write_text(
        "Maintainer documentation added with the merge.\n", encoding="utf-8"
    )
    _commit(merged, "Squash merged uploaded answer")
    _git(merged, "push", "origin", "main")

    run_update(SimpleNamespace(repository="owner/repo"))

    assert "later review answer" in deck.read_text(encoding="utf-8")
    assert (source / "README.md").exists()
    assert _git(source, "status", "--short").stdout == ""
    db = SyncState.open(collection)
    try:
        assert db.get_collab_operation("collab/owner/repo") is None
    finally:
        db.close()


def test_submit_auth_failure_reports_simple_retry(collab_world, monkeypatch):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally committed answer"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("GitHub CLI is not authenticated. Run: gh auth login")
        ),
    )

    with pytest.raises(ValueError) as error:
        run_submit(SimpleNamespace(repository="owner/repo", title="Local change"))

    message = str(error.value)
    assert "was committed locally" in message
    assert "Nothing was uploaded" in message
    assert "Retrying is safe: ankiops collab submit owner/repo" in message
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "ready"
        assert operation["requested_title"] == "Local change"
    finally:
        db.close()
    assert _git(source, "branch", "--list", "ankiops/*").stdout.strip() == (
        "* ankiops/journal"
    )


def test_submit_bare_retry_preserves_title_after_github_setup_failure(
    collab_world, tmp_path, monkeypatch
):
    collection, source, _remote = collab_world
    publish = tmp_path / "setup-retry-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "retry title answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    attempts = 0

    def fail_setup_once(*_args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("simulated GitHub setup failure")
        return "contributor/repo", "contributor"

    created_prs = []

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return "https://github.test/owner/repo/pull/title-retry"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target", fail_setup_once
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)

    with pytest.raises(ValueError, match="GitHub setup did not finish"):
        run_submit(
            SimpleNamespace(repository="owner/repo", title="Preserve requested title")
        )

    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    assert created_prs[0]["title"] == "Preserve requested title"
    branch = created_prs[0]["head"].split(":", 1)[1]
    assert (
        _git(publish, "log", "-1", "--format=%s", branch).stdout.strip()
        == "Preserve requested title"
    )
    state = SyncState.open(collection)
    try:
        operation = state.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["requested_title"] is None
    finally:
        state.close()


def test_status_reports_local_state_when_fetch_fails(collab_world, monkeypatch, capsys):
    _collection, source, _remote = collab_world
    (source / "Draft.md").write_text("draft\n", encoding="utf-8")
    monkeypatch.setattr(
        GitRepository,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "fetch"])
        ),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Working tree: 1 changed path" in output
    assert "Upstream: unavailable" in output
    assert "collection changes" not in output.lower()
    assert "Retry status: ankiops collab status owner/repo" in output


def test_status_never_writes_untracked_content_to_git_objects(collab_world, capsys):
    _collection, source, _remote = collab_world
    secret_content = "untracked status secret that must never enter Git objects\n"
    secret = source / "Untracked Secret.txt"
    secret.write_text(secret_content, encoding="utf-8")
    secret_oid = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=source,
        input=secret_content,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert _git(source, "cat-file", "-e", secret_oid, check=False).returncode != 0
    status_before = _git(source, "status", "--porcelain=v1", "-z").stdout

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Untracked Secret.txt" in output
    assert _git(source, "cat-file", "-e", secret_oid, check=False).returncode != 0
    assert _git(source, "status", "--porcelain=v1", "-z").stdout == status_before


def test_status_reports_committed_local_contribution(collab_world, capsys):
    _collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "locally committed answer"
        ),
        encoding="utf-8",
    )
    _commit(source, "Local contribution")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert output.startswith("owner/repo\n")
    assert "Working tree: clean" in output
    assert "Local contribution: 1 change ready to submit" in output
    assert "Upstream: up to date" in output
    assert "Anki:" not in output
    assert "Apply to Anki" not in output
    assert "Submit contribution: ankiops collab submit owner/repo" in output


def test_status_separates_multiple_source_blocks(collab_world, capsys):
    collection, _source, remote = collab_world
    second = collection / "collab" / "second-owner" / "second-repo"
    second.parent.mkdir(parents=True)
    _git(second.parent, "clone", "--origin", "upstream", str(remote), str(second))
    _configure(second, "Second Contributor")
    _git(second, "checkout", "-b", "ankiops/journal", "upstream/main")
    _git(second, "branch", "--unset-upstream")
    _git(second, "update-ref", "refs/ankiops/integrated", "upstream/main")

    run_status(SimpleNamespace(repository=None))

    output = capsys.readouterr().out
    assert "Upstream: up to date\n\nsecond-owner/second-repo" in output
    assert output.count("Working tree: clean") == 2


def test_status_reports_available_upstream_update(collab_world, tmp_path, capsys):
    _collection, _source, remote = collab_world
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Local contribution: none" in output
    assert "Upstream: 1 update available" in output
    assert "Integrate upstream: ankiops collab update owner/repo" in output
    assert "Submit contribution:" not in output


def test_status_reports_diverged_contributions(collab_world, tmp_path, capsys):
    _collection, source, remote = collab_world
    (source / "Local.md").write_text("local\n", encoding="utf-8")
    _commit(source, "Local contribution")
    _upstream_edit(tmp_path, remote, "upstream answer", "upstream correction")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Local contribution: 1 change ready to submit" in output
    assert "Upstream: 1 update available; update before submitting" in output
    assert "Integrate upstream: ankiops collab update owner/repo" in output
    assert "Submit contribution:" not in output


def test_status_reports_contribution_accepted_upstream(collab_world, tmp_path, capsys):
    collection, source, remote = collab_world
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
        db.save_collab_operation(
            "collab/owner/repo",
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

    output = capsys.readouterr().out
    assert "Local contribution: none" in output
    assert "Upstream: up to date" in output
    assert "Submission: merged; update to integrate and clean up" in output
    assert "Integrate merge: ankiops collab update owner/repo" in output
    assert "Review pull request:" not in output


def test_status_omits_unrelated_collection_changes(collab_world, capsys):
    collection, _source, _remote = collab_world
    private_deck = collection / "Private.md"
    private_deck.write_text("private edit\n", encoding="utf-8")
    generated_config = collection / "note_types" / "Generated.yaml"
    generated_config.parent.mkdir()
    generated_config.write_text("generated\n", encoding="utf-8")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "collection changes" not in output.lower()
    assert "Private.md" not in output
    assert "Generated.yaml" not in output


def test_status_prints_unicode_collab_paths_without_git_quoting(collab_world, capsys):
    _collection, source, _remote = collab_world
    unicode_name = "Déck Ω — punctuation!.md"
    (source / unicode_name).write_text("draft\n", encoding="utf-8")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert unicode_name in output
    assert "\\303" not in output


@pytest.mark.parametrize(
    ("state", "submission_status"),
    [
        ("push_failed", "not uploaded"),
        ("pr_failed", "uploaded; pull request not created"),
    ],
)
def test_status_gives_submit_retry_as_next_command(
    collab_world, capsys, state, submission_status
):
    collection, source, _remote = collab_world
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
        db.save_collab_operation(
            "collab/owner/repo",
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

    output = capsys.readouterr().out
    assert f"Submission: {submission_status}" in output
    assert "Retry submission: ankiops collab submit owner/repo" in output


def test_submit_updates_only_the_title_of_an_unchanged_open_pull_request(
    collab_world, tmp_path, monkeypatch, capsys
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "title-only-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []
    updated_prs = []
    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return "https://github.com/owner/repo/pull/23"

    def update_pr(_host, url, *, title):
        updated_prs.append((url, title))

    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.update_pr", update_pr)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN"),
    )

    run_submit(SimpleNamespace(repository="owner/repo", title="Initial title"))
    submitted_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    submitted_refs = _git(
        publish, "for-each-ref", "--format=%(objectname) %(refname)"
    ).stdout
    capsys.readouterr()

    run_submit(SimpleNamespace(repository="owner/repo", title="Clearer title"))

    assert updated_prs == [("https://github.com/owner/repo/pull/23", "Clearer title")]
    assert len(created_prs) == 1
    assert _git(source, "rev-parse", "HEAD").stdout.strip() == submitted_head
    assert (
        _git(publish, "for-each-ref", "--format=%(objectname) %(refname)").stdout
        == submitted_refs
    )
    output = capsys.readouterr().out
    assert "pull request updated" in output
    assert "Title: Clearer title" in output
    assert "pull request already open" not in output


@pytest.mark.parametrize(
    "commit_revision",
    [False, True],
    ids=["uncommitted-edit", "committed-edit"],
)
def test_status_recommends_updating_an_open_pull_request_after_local_edits(
    collab_world, monkeypatch, capsys, commit_revision
):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Submitted contribution")
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "open-with-edits",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/open-with-edits",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/24",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(state="OPEN"),
    )
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("submitted answer", "revised answer"),
        encoding="utf-8",
    )
    if commit_revision:
        _commit(source, "Revise contribution")

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Submission: pull request open; local changes not uploaded" in output
    assert "Update pull request: ankiops collab submit owner/repo" in output
    assert "Submit contribution:" not in output


@pytest.mark.parametrize(
    "mismatch",
    [
        "head-commit",
        "head-branch",
        "head-owner",
        "publish-repository",
        "unparsed-publish-repository",
    ],
)
def test_status_reports_when_an_open_pull_request_changed_on_github(
    collab_world, monkeypatch, capsys, mismatch
):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Submitted contribution")
    publish_repository = "other-repo" if mismatch == "publish-repository" else "repo"
    publish_url = (
        "custom://unverified/contributor/repo.git"
        if mismatch == "unparsed-publish-repository"
        else f"https://github.com/contributor/{publish_repository}.git"
    )
    _git(
        source,
        "remote",
        "add",
        "publish",
        publish_url,
    )
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "github-edited",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/github-edited",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/25",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="OPEN",
            head_owner="attacker" if mismatch == "head-owner" else "contributor",
            head_repository="contributor/repo",
            head_branch=(
                "ankiops/repointed"
                if mismatch == "head-branch"
                else "ankiops/github-edited"
            ),
            head_sha=(
                "maintainer-edited-head" if mismatch == "head-commit" else prepared_head
            ),
        ),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert (
        "Submission: pull request open; changed on GitHub since your last upload"
        in output
    )
    assert "Local contribution: 1 change uploaded" not in output
    assert "Update pull request: ankiops collab submit owner/repo" not in output


def test_status_accepts_an_alternate_named_fork_as_the_pull_request_head(
    collab_world, monkeypatch, capsys
):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Submitted contribution")
    _git(
        source,
        "remote",
        "add",
        "publish",
        "https://github.com/contributor/repo-ankiops.git",
    )
    state = SyncState.open(collection)
    try:
        state.save_collab_operation(
            "collab/owner/repo",
            "alternate-fork",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/alternate-fork",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/alternate-fork",
        )
    finally:
        state.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="OPEN",
            head_owner="contributor",
            head_repository="contributor/repo-ankiops",
            head_branch="ankiops/alternate-fork",
            head_sha=prepared_head,
        ),
    )

    run_status(SimpleNamespace(repository="owner/repo"))

    output = capsys.readouterr().out
    assert "Local contribution: 1 change uploaded" in output
    assert "Submission: pull request open" in output
    assert "changed on GitHub since your last upload" not in output


def test_submit_never_overwrites_an_open_pull_request_changed_on_github(
    collab_world, tmp_path, monkeypatch
):
    _collection, source, _remote = collab_world
    publish = tmp_path / "maintainer-edited-open-pr.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first submitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.create_pr",
        lambda *_args, **_kwargs: "https://github.com/owner/repo/pull/26",
    )
    run_submit(SimpleNamespace(repository="owner/repo", title="First revision"))
    branch = _git(
        publish, "for-each-ref", "--format=%(refname:short)", "refs/heads/ankiops/*"
    ).stdout.strip()

    maintainer = tmp_path / "maintainer-edited-pr"
    _git(tmp_path, "clone", str(publish), str(maintainer))
    _configure(maintainer, "Maintainer")
    _git(maintainer, "checkout", branch)
    maintainer_deck = maintainer / "Deck.md"
    maintainer_deck.write_text(
        maintainer_deck.read_text(encoding="utf-8").replace(
            "first submitted answer", "maintainer-edited answer"
        ),
        encoding="utf-8",
    )
    maintainer_head = _commit(maintainer, "Maintainer edits contribution")
    _git(maintainer, "push", "origin", branch)

    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first submitted answer", "contributor second revision"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="OPEN",
            head_owner="contributor",
            head_branch=branch,
            head_sha=maintainer_head,
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.login", lambda *_args: "contributor"
    )
    monkeypatch.setattr(
        GitRepository,
        "push_force_with_lease",
        lambda *_args: pytest.fail("GitHub-edited PR head must not be overwritten"),
    )

    with pytest.raises(ValueError, match="changed on GitHub.*Nothing was sent"):
        run_submit(SimpleNamespace(repository="owner/repo", title="Second revision"))

    assert _git(publish, "rev-parse", branch).stdout.strip() == maintainer_head
    assert "contributor second revision" in deck.read_text(encoding="utf-8")


def test_submit_unchanged_or_title_only_reports_a_github_edited_head(
    collab_world, monkeypatch, capsys
):
    collection, source, _remote = collab_world
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace("upstream answer", "submitted answer"),
        encoding="utf-8",
    )
    prepared_head = _commit(source, "Submitted contribution")
    _git(
        source,
        "remote",
        "add",
        "publish",
        "https://github.com/contributor/repo.git",
    )
    db = SyncState.open(collection)
    try:
        db.save_collab_operation(
            "collab/owner/repo",
            "title-after-github-edit",
            "submit",
            "pr_open",
            prepared_head=prepared_head,
            upstream_tree=_git(
                source, "rev-parse", "upstream/main^{tree}"
            ).stdout.strip(),
            publish_branch="ankiops/title-after-github-edit",
            pushed_sha=prepared_head,
            pr_url="https://github.com/owner/repo/pull/27",
        )
    finally:
        db.close()
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.pull_request",
        lambda *_args: SimpleNamespace(
            state="OPEN",
            head_owner="contributor",
            head_branch="ankiops/title-after-github-edit",
            head_sha="github-edited-head",
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.update_pr",
        lambda *_args, **_kwargs: pytest.fail(
            "title update must not disguise a GitHub-edited head"
        ),
    )

    run_submit(SimpleNamespace(repository="owner/repo", title=None))

    output = capsys.readouterr().out
    assert "pull request changed on GitHub" in output
    assert "pull request already open" not in output
    with pytest.raises(ValueError, match="changed on GitHub.*Nothing was sent"):
        run_submit(SimpleNamespace(repository="owner/repo", title="New title"))


@pytest.mark.parametrize(
    "remote_updated_before_interruption",
    [False, True],
    ids=["before-push", "after-remote-update"],
)
def test_submit_retry_around_replacement_push_keeps_the_existing_pull_request(
    collab_world, tmp_path, monkeypatch, remote_updated_before_interruption
):
    collection, source, _remote = collab_world
    publish = tmp_path / "interrupted-revision-fork.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    deck = source / "Deck.md"
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "upstream answer", "first submitted answer"
        ),
        encoding="utf-8",
    )
    original_set_remote = GitRepository.set_remote

    def local_publish(source_git, name, _url):
        original_set_remote(source_git, name, str(publish))

    created_prs = []

    def create_pr(*_args, **kwargs):
        created_prs.append(kwargs)
        return "https://github.test/owner/repo/pull/existing"

    monkeypatch.setattr(GitRepository, "set_remote", local_publish)
    monkeypatch.setattr(
        "ankiops.collab.commands.GitHubHost.publish_target",
        lambda *_args: ("contributor/repo", "contributor"),
    )
    monkeypatch.setattr("ankiops.collab.commands.GitHubHost.create_pr", create_pr)
    args = SimpleNamespace(repository="owner/repo", title="Improve answer")

    run_submit(args)
    first_uploaded = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    deck.write_text(
        deck.read_text(encoding="utf-8").replace(
            "first submitted answer", "revised submitted answer"
        ),
        encoding="utf-8",
    )

    original_force_push = GitRepository.push_force_with_lease

    def interrupt_replacement_push(source_git, remote, commit, branch, expected_sha):
        if remote_updated_before_interruption:
            original_force_push(source_git, remote, commit, branch, expected_sha)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        GitRepository,
        "push_force_with_lease",
        interrupt_replacement_push,
    )
    with pytest.raises(KeyboardInterrupt):
        run_submit(args)
    monkeypatch.setattr(GitRepository, "push_force_with_lease", original_force_push)

    run_submit(args)

    second_uploaded = _git(
        publish, "for-each-ref", "--format=%(objectname)", "refs/heads/ankiops/*"
    ).stdout.strip()
    assert second_uploaded != first_uploaded
    assert (
        "revised submitted answer"
        in _git(publish, "show", f"{second_uploaded}:Deck.md").stdout
    )
    assert len(created_prs) == 1
    db = SyncState.open(collection)
    try:
        operation = db.get_collab_operation("collab/owner/repo")
        assert operation is not None
        assert operation["state"] == "pr_open"
        assert operation["pr_url"] == ("https://github.test/owner/repo/pull/existing")
        assert operation["pushed_sha"] == second_uploaded
    finally:
        db.close()

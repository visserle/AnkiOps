from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pytest

from ankiops.deck_sources import DeckSource
from ankiops.git import CollectionGit
from ankiops.markdown import NOTE_SEPARATOR
from ankiops.shared import commands as shared_commands
from ankiops.shared import run_create, run_list, run_submit, run_update
from ankiops.shared.commands import _parse_slug
from ankiops.shared.create import _unlink_created_source_files
from ankiops.shared.hosting import ensure_create_repo, open_pr_if_possible
from tests.support.deck_files import DeckFileHarness


def _setup_collection(tmp_path):
    DeckFileHarness().eject_default_note_types(tmp_path / "note_types")
    (tmp_path / "media").mkdir()
    return tmp_path


def _init_git_repo(collection_dir):
    subprocess.run(["git", "init"], cwd=collection_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=collection_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=collection_dir,
        check=True,
    )


def _git_head(collection_dir):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _git_status(collection_dir):
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _commit_all(collection_dir, message="root"):
    subprocess.run(["git", "add", "-A", "."], cwd=collection_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=collection_dir,
        check=True,
        capture_output=True,
    )


def _setup_shared_source_with_remote(tmp_path):
    collection_dir = _setup_collection(tmp_path / "collection")
    shared_root = collection_dir / "shared" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(shared_root / "note_types")
    (shared_root / "Deck.md").write_text(
        "<!-- note_key: key-1 -->\n"
        "<!-- note_type: shared/owner/repo/AnkiOpsQA -->\n"
        "Q: local\nA: deck\n",
        encoding="utf-8",
    )
    (collection_dir / "Private.md").write_text("original\n", encoding="utf-8")
    (collection_dir / "Unstaged.md").write_text("original\n", encoding="utf-8")
    _init_git_repo(collection_dir)
    _commit_all(collection_dir)

    source = DeckSource.shared(collection_dir, "owner", "repo")
    repo = CollectionGit(collection_dir)
    split_sha = repo.split_subtree(source)
    repo.rejoin_subtree(
        source,
        split_sha,
        "AnkiOps: initialize subtree history for shared/owner/repo",
    )
    remote = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", remote],
        check=True,
        capture_output=True,
    )
    repo.push_ref(str(remote), split_sha, "refs/heads/main")
    return collection_dir, source, repo, remote


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("owner/repo", ("owner", "repo")),
        ("owner-name/repo-name", ("owner-name", "repo-name")),
        ("o/r", ("o", "r")),
    ],
)
def test_parse_slug_accepts_safe_owner_repo(slug, expected):
    assert _parse_slug(slug) == expected


@pytest.mark.parametrize(
    "slug",
    [
        "owner/Segeln_",
        "owner/Segeln.",
        "owner/Segeln+",
        "owner/-Segeln",
        "owner/Segeln-",
        "owner_name/Segeln",
    ],
)
def test_parse_slug_rejects_unsafe_owner_repo(slug):
    with pytest.raises(ValueError, match="Invalid GitHub repo slug"):
        _parse_slug(slug)


def test_list_logs_clickable_paths_as_rich_markup(tmp_path, monkeypatch, caplog):
    shared_root = tmp_path / "shared" / "[owner]" / "repo"
    shared_root.mkdir(parents=True)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: tmp_path,
    )

    with caplog.at_level(logging.INFO, logger="ankiops.shared.commands"):
        run_list(SimpleNamespace())

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert getattr(record, "markup") is True
    assert "shared/\\[owner]/repo" in record.getMessage()
    assert f"[link={shared_root.resolve().as_uri()}]repo[/link]" in record.getMessage()


def test_create_unsynced_deck_moves_files_media_and_note_types(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: ![img](media/pic.png)\n",
        encoding="utf-8",
    )
    (collection_dir / "media" / "pic.png").write_bytes(b"img")
    _init_git_repo(collection_dir)
    _commit_all(collection_dir)
    pushed = []
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.git.CollectionGit.push_ref",
        lambda repo, url, source_ref, target_ref, *, check=True: pushed.append(
            (repo.collection_dir, url, source_ref, target_ref, check)
        ),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    shared_root = collection_dir / "shared" / "owner" / "repo"
    assert not deck.exists()
    assert (shared_root / "Deck.md").exists()
    content = (shared_root / "Deck.md").read_text(encoding="utf-8")
    assert "<!-- note_type: shared/owner/repo/AnkiOpsQA -->" in content
    assert (shared_root / "media" / "pic.png").read_bytes() == b"img"
    assert (shared_root / "note_types" / "AnkiOpsQA").is_dir()
    assert (shared_root / "note_types" / "AnkiOpsStyling.css").exists()
    assert (shared_root / "note_types" / "SyntaxHighlighting.css").exists()
    DeckFileHarness().load_note_types(shared_root / "note_types")
    assert _git_status(collection_dir) == ""
    assert pushed[0][1] == "https://github.com/owner/repo.git"
    assert pushed[0][3] == "main"
    message = subprocess.run(
        ["git", "show", "-s", "--format=%B", "HEAD"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "AnkiOps: initialize subtree history for shared/owner/repo" in message
    assert "git-subtree-dir: shared/owner/repo" in message
    branches = subprocess.run(
        ["git", "branch", "--list", "ankiops-shared-owner-repo-*"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert branches.stdout == ""


def test_create_missing_repo_blocks_before_anki_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("<!-- note_key: key-1 -->\nQ: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _init_git_repo(collection_dir)
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("GitHub repository does not exist")
        ),
    )

    with pytest.raises(ValueError, match="GitHub repository does not exist"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "shared").exists()


def test_create_rejects_unsafe_repo_slug_before_git_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("Q: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    with pytest.raises(ValueError, match="Invalid GitHub repo slug"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/Segeln_"))

    assert deck.exists()
    assert not (collection_dir / "shared").exists()


def test_create_missing_referenced_media_blocks_before_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: ![img](media/missing.png)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _init_git_repo(collection_dir)

    with pytest.raises(ValueError, match="referenced media file\\(s\\) missing"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "shared").exists()


def test_create_rejects_missing_note_keys_before_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("Q: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _init_git_repo(collection_dir)

    with pytest.raises(ValueError) as excinfo:
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    message = str(excinfo.value)
    assert message == (
        "Missing note_keys for 1 note. "
        "note_keys are stable IDs AnkiOps needs to match notes across "
        "collections without duplicates. "
        "Fix: run 'ankiops fa' to assign them."
    )
    assert "explicit note_key" not in message
    assert "Deck.md note" not in message
    assert deck.exists()
    assert not (collection_dir / "shared").exists()


def test_create_commit_failure_keeps_source_file(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    subprocess.run(["git", "init"], cwd=collection_dir, check=True, capture_output=True)
    deck = collection_dir / "Deck.md"
    original = "<!-- note_key: key-1 -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.CollectionGit.commit_create_move",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("commit failed")),
    )

    with pytest.raises(ValueError, match="commit failed"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "shared").exists()


def test_create_dirty_index_blocks_before_file_mutations_and_preserves_stage(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    subprocess.run(["git", "init"], cwd=collection_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=collection_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=collection_dir,
        check=True,
    )
    deck = collection_dir / "Deck.md"
    original = "<!-- note_key: key-1 -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    other = collection_dir / "Other.md"
    other.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=collection_dir, check=True)
    subprocess.run(["git", "commit", "-m", "root"], cwd=collection_dir, check=True)

    other.write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "Other.md"], cwd=collection_dir, check=True)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.shared.create._write_create_files",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )
    monkeypatch.setattr(
        "ankiops.shared.create._cleanup_failed_create",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected cleanup")),
    )

    with pytest.raises(ValueError, match="clean git index"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == "M  Other.md\n"
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "shared").exists()


def test_create_subtree_split_failure_rolls_back_create_commit(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    _init_git_repo(collection_dir)
    deck = collection_dir / "Deck.md"
    original = "<!-- note_key: key-1 -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=collection_dir, check=True)
    subprocess.run(["git", "commit", "-m", "root"], cwd=collection_dir, check=True)
    initial_head = _git_head(collection_dir)

    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.CollectionGit.split_subtree",
        lambda *_args: (_ for _ in ()).throw(ValueError("split failed")),
    )

    with pytest.raises(ValueError, match="split failed"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert _git_head(collection_dir) == initial_head
    assert _git_status(collection_dir) == ""
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "shared").exists()


def test_create_push_failure_rolls_back_create_commit_and_temp_branch(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    _init_git_repo(collection_dir)
    deck = collection_dir / "Deck.md"
    original = "<!-- note_key: key-1 -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=collection_dir, check=True)
    subprocess.run(["git", "commit", "-m", "root"], cwd=collection_dir, check=True)
    initial_head = _git_head(collection_dir)
    branch = "ankiops-test-branch"

    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.create.ensure_create_repo",
        lambda *_args, **_kwargs: None,
    )

    def fake_branch(repo, _source, _commit_sha):
        subprocess.run(["git", "branch", branch], cwd=repo.collection_dir, check=True)
        return branch

    monkeypatch.setattr(
        "ankiops.shared.create.CollectionGit.create_temp_branch",
        fake_branch,
    )

    monkeypatch.setattr(
        "ankiops.shared.create.CollectionGit.push_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("push failed")),
    )

    with pytest.raises(ValueError, match="push failed"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert _git_head(collection_dir) == initial_head
    assert _git_status(collection_dir) == ""
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "shared").exists()
    branches = subprocess.run(
        ["git", "branch", "--list", branch],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert branches.stdout == ""


def test_git_commit_paths_ignores_missing_untracked_source_path(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    target = tmp_path / "shared" / "owner" / "repo" / "Segeln.md"
    target.parent.mkdir(parents=True)
    target.write_text("Q: moved\nA: deck\n", encoding="utf-8")

    committed = CollectionGit(tmp_path).commit_paths(
        [target, tmp_path / "Segeln.md"],
        "create",
    )

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == ""
    assert committed is True
    assert CollectionGit(tmp_path).commit_paths([target], "no-op") is False


def test_git_commit_create_stages_source_removal_before_unlink(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "Deck.md"
    source.write_text("Q: root\nA: deck\n", encoding="utf-8")
    subprocess.run(["git", "add", "Deck.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "root"], cwd=tmp_path, check=True)

    target = tmp_path / "shared" / "owner" / "repo" / "Deck.md"
    target.parent.mkdir(parents=True)
    target.write_text("Q: shared\nA: deck\n", encoding="utf-8")
    plan = SimpleNamespace(files=[SimpleNamespace(source_path=source)])

    CollectionGit(tmp_path).commit_create_move(
        touched_paths=[target.parent, target, source],
        source_paths=[source],
        message="create",
    )

    assert source.exists()
    show = subprocess.run(
        ["git", "show", "--name-status", "--format=", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "D\tDeck.md" in show.stdout
    assert "A\tshared/owner/repo/Deck.md" in show.stdout

    _unlink_created_source_files(plan)

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == ""


def test_rejoin_subtree_preserves_private_changes_and_records_metadata(tmp_path):
    _init_git_repo(tmp_path)
    source = DeckSource.shared(tmp_path, "owner", "repo")
    source.root.mkdir(parents=True)
    (source.root / "Deck.md").write_text("Q: shared\nA: deck\n", encoding="utf-8")
    private = tmp_path / "Private.md"
    private.write_text("original\n", encoding="utf-8")
    unstaged = tmp_path / "Unstaged.md"
    unstaged.write_text("original\n", encoding="utf-8")
    _commit_all(tmp_path)

    repo = CollectionGit(tmp_path)
    split_sha = repo.split_subtree(source)
    old_head = _git_head(tmp_path)
    old_tree = repo.run(["rev-parse", "HEAD^{tree}"]).stdout.strip()

    private.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "Private.md"], cwd=tmp_path, check=True)
    unstaged.write_text("unstaged\n", encoding="utf-8")
    old_status = _git_status(tmp_path)

    repo.rejoin_subtree(
        source,
        split_sha,
        "AnkiOps: initialize subtree history for shared/owner/repo",
    )

    new_head = _git_head(tmp_path)
    assert new_head != old_head
    assert repo.run(["rev-parse", "HEAD^{tree}"]).stdout.strip() == old_tree
    assert repo.run(["rev-parse", "HEAD^1"]).stdout.strip() == old_head
    assert repo.run(["rev-parse", "HEAD^2"]).stdout.strip() == split_sha
    message = repo.run(["show", "-s", "--format=%B", "HEAD"]).stdout
    assert "git-subtree-dir: shared/owner/repo" in message
    assert f"git-subtree-mainline: {old_head}" in message
    assert f"git-subtree-split: {split_sha}" in message
    assert _git_status(tmp_path) == old_status


def test_rejoin_subtree_rejects_a_split_with_different_content(tmp_path):
    _init_git_repo(tmp_path)
    source = DeckSource.shared(tmp_path, "owner", "repo")
    source.root.mkdir(parents=True)
    deck = source.root / "Deck.md"
    deck.write_text("Q: original\nA: deck\n", encoding="utf-8")
    _commit_all(tmp_path)
    repo = CollectionGit(tmp_path)
    stale_split = repo.split_subtree(source)

    deck.write_text("Q: changed\nA: deck\n", encoding="utf-8")
    repo.commit_paths([deck], "change shared deck")
    current_head = _git_head(tmp_path)

    with pytest.raises(ValueError, match="split tree does not match"):
        repo.rejoin_subtree(source, stale_split, "rejoin")

    assert _git_head(tmp_path) == current_head


def test_ensure_create_repo_without_gh_prints_manual_create_command(
    tmp_path,
    monkeypatch,
):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    monkeypatch.setattr(
        "ankiops.shared.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr("ankiops.shared.hosting.shutil.which", lambda _name: None)

    with pytest.raises(ValueError, match="gh repo create owner/repo --private"):
        ensure_create_repo(CollectionGit(tmp_path), source, public=False)


def test_ensure_create_repo_creates_missing_private_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr(
        "ankiops.shared.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr(
        "ankiops.shared.hosting.shutil.which",
        lambda _name: "/bin/gh",
    )
    monkeypatch.setattr(
        "ankiops.shared.hosting._create_github_repo",
        lambda repo, source, *, public: calls.append(
            (repo.collection_dir, source.github_slug, public)
        ),
    )

    ensure_create_repo(CollectionGit(tmp_path), source, public=False)

    assert calls == [(tmp_path, "owner/repo", False)]


def test_ensure_create_repo_creates_missing_public_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr(
        "ankiops.shared.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr(
        "ankiops.shared.hosting.shutil.which",
        lambda _name: "/bin/gh",
    )
    monkeypatch.setattr(
        "ankiops.shared.hosting._create_github_repo",
        lambda repo, source, *, public: calls.append(
            (repo.collection_dir, source.github_slug, public)
        ),
    )

    ensure_create_repo(CollectionGit(tmp_path), source, public=True)

    assert calls == [(tmp_path, "owner/repo", True)]


def test_open_pr_uses_explicit_title_and_reports_success(tmp_path, monkeypatch):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    repo = CollectionGit(tmp_path)
    monkeypatch.setattr("ankiops.shared.hosting.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        "ankiops.git.CollectionGit.push_ref",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "https://example.test/pr/1\n", "")

    monkeypatch.setattr("ankiops.shared.hosting.subprocess.run", fake_run)

    pushed = open_pr_if_possible(
        repo,
        source,
        "submission-branch",
        title="Clarify shared history",
    )

    assert pushed is True
    assert calls[0][0] == [
        "gh",
        "pr",
        "create",
        "--repo",
        "owner/repo",
        "--head",
        "submission-branch",
        "--base",
        "main",
        "--title",
        "Clarify shared history",
        "--body",
        "Submitted by AnkiOps from shared/owner/repo.",
    ]


def test_create_rejects_duplicate_note_keys(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        NOTE_SEPARATOR.join(
            [
                "<!-- note_key: duplicate -->\nQ: one\nA: one",
                "<!-- note_key: duplicate -->\nQ: two\nA: two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _init_git_repo(collection_dir)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with pytest.raises(ValueError, match="Duplicate note_key"):
        run_create(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "shared").exists()


def test_submit_rejects_keyless_notes_without_mutating_files(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    shared_root = collection_dir / "shared" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(shared_root / "note_types")
    deck = shared_root / "Deck.md"
    original = "<!-- note_type: shared/owner/repo/AnkiOpsQA -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    _init_git_repo(collection_dir)
    _commit_all(collection_dir)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with pytest.raises(ValueError) as excinfo:
        run_submit(SimpleNamespace(repo="owner/repo"))

    message = str(excinfo.value)
    assert message == (
        "Missing note_keys for 1 note. "
        "note_keys are stable IDs AnkiOps needs to match notes across "
        "collections without duplicates. "
        "Fix: run 'ankiops fa' to assign them."
    )
    assert "explicit note_key" not in message
    assert "Deck.md note" not in message
    assert deck.read_text(encoding="utf-8") == original
    assert _git_status(collection_dir) == ""


def test_submit_with_no_shared_changes_creates_no_branch_or_pr(
    tmp_path,
    monkeypatch,
    caplog,
):
    collection_dir, _source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    initial_head = _git_head(collection_dir)

    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.open_pr_if_possible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected PR")
        ),
    )

    with caplog.at_level(logging.INFO, logger="ankiops.shared.commands"):
        run_submit(SimpleNamespace(repo="owner/repo", message=None))

    assert _git_head(collection_dir) == initial_head
    assert "No shared changes to submit for owner/repo." in caplog.text
    assert repo.run(["branch", "--list", "ankiops-shared-*"]).stdout == ""


def test_submit_requires_explicit_commit_for_dirty_shared_changes(
    tmp_path,
    monkeypatch,
):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    deck.write_text(deck.read_text() + "E: clarified\n", encoding="utf-8")
    initial_head = repo.head()
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with pytest.raises(ValueError, match="--commit"):
        run_submit(SimpleNamespace(repo="owner/repo", message=None, commit=False))

    assert repo.head() == initial_head
    assert "Deck.md" in _git_status(collection_dir)


def test_shared_status_reports_changes_remote_state_and_submit_action(
    tmp_path,
    monkeypatch,
    caplog,
):
    collection_dir, source, _repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    deck.write_text(deck.read_text() + "E: clarified\n", encoding="utf-8")
    (collection_dir / "Private.md").write_text("private draft\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with caplog.at_level(logging.INFO, logger="ankiops.shared.commands"):
        shared_commands.run_status(SimpleNamespace(repo="owner/repo"))

    assert "Shared changes: 1" in caplog.text
    assert "Deck.md" in caplog.text
    assert "Private changes: 1 (preserved)" in caplog.text
    assert "Private.md" in caplog.text
    assert "Committed shared state matches GitHub main." in caplog.text
    assert "Submit: blocked" in caplog.text
    assert "--commit" in caplog.text


def test_shared_status_reports_precommitted_changes_will_open_pr(
    tmp_path,
    monkeypatch,
    caplog,
):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    deck.write_text(deck.read_text() + "E: precommitted\n", encoding="utf-8")
    repo.commit_paths([deck], "Clarify shared deck")
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with caplog.at_level(logging.INFO, logger="ankiops.shared.commands"):
        shared_commands.run_status(SimpleNamespace(repo="owner/repo"))

    assert "Shared changes: 0" in caplog.text
    assert "has changes not on GitHub main" in caplog.text
    assert "Submit: will open a pull request." in caplog.text


def test_submit_preserves_private_changes_and_uses_custom_message(
    tmp_path,
    monkeypatch,
):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    private = collection_dir / "Private.md"
    unstaged = collection_dir / "Unstaged.md"

    deck.write_text(deck.read_text() + "E: clarified\n", encoding="utf-8")
    private.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "Private.md"], cwd=collection_dir, check=True)
    unstaged.write_text("unstaged\n", encoding="utf-8")
    expected_private_status = "M  Private.md\n M Unstaged.md\n"

    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    submitted = []

    def fake_open_pr(repo, source, branch, *, title):
        branch_sha = repo.run(["rev-parse", branch]).stdout.strip()
        submitted.append((branch, title, branch_sha))
        return True

    monkeypatch.setattr("ankiops.shared.commands.open_pr_if_possible", fake_open_pr)

    run_submit(
        SimpleNamespace(
            repo="owner/repo",
            message="Clarify shared history",
            commit=True,
        )
    )

    assert submitted[0][1] == "Clarify shared history"
    remote_head = repo.run(["rev-parse", "FETCH_HEAD"]).stdout.strip()
    assert repo.is_ancestor(remote_head, submitted[0][2])
    assert _git_status(collection_dir) == expected_private_status
    assert repo.run(["branch", "--list", submitted[0][0]]).stdout == ""
    subjects = repo.run(
        ["log", "--first-parent", "--format=%s", "-2"]
    ).stdout.splitlines()
    assert subjects == [
        "AnkiOps: record submission history for shared/owner/repo",
        "AnkiOps: Clarify shared history",
    ]


def test_submit_retains_local_branch_when_push_fails(tmp_path, monkeypatch):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    deck.write_text(deck.read_text() + "E: changed\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    submitted = []

    def fake_open_pr(_repo, _source, branch, *, title):
        submitted.append((branch, title))
        return False

    monkeypatch.setattr("ankiops.shared.commands.open_pr_if_possible", fake_open_pr)

    run_submit(SimpleNamespace(repo="owner/repo", message=None, commit=True))

    branch, title = submitted[0]
    assert title == "Update shared deck owner/repo"
    assert repo.run(["branch", "--list", branch]).stdout.strip().endswith(branch)


def test_submit_is_noop_when_only_remote_has_advanced(tmp_path, monkeypatch):
    collection_dir, _source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    collaborator = tmp_path / "collaborator"
    subprocess.run(
        ["git", "clone", remote, collaborator],
        check=True,
        capture_output=True,
    )
    _configure = [
        ("user.name", "Collaborator"),
        ("user.email", "collaborator@example.invalid"),
    ]
    for key, value in _configure:
        subprocess.run(
            ["git", "config", key, value],
            cwd=collaborator,
            check=True,
        )
    deck = collaborator / "Deck.md"
    deck.write_text(deck.read_text() + "E: remote\n", encoding="utf-8")
    subprocess.run(["git", "add", "Deck.md"], cwd=collaborator, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Remote edit"],
        cwd=collaborator,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=collaborator,
        check=True,
        capture_output=True,
    )
    initial_head = repo.head()
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.open_pr_if_possible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected PR")
        ),
    )

    run_submit(SimpleNamespace(repo="owner/repo", message=None))

    assert repo.head() == initial_head


def test_submit_is_noop_when_divergent_history_has_same_tree(tmp_path, monkeypatch):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    original = deck.read_text(encoding="utf-8")
    deck.write_text(original + "E: temporary\n", encoding="utf-8")
    repo.commit_paths([deck], "Temporary shared edit")
    deck.write_text(original, encoding="utf-8")
    repo.commit_paths([deck], "Revert temporary shared edit")
    initial_head = repo.head()
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.open_pr_if_possible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected PR")
        ),
    )

    run_submit(SimpleNamespace(repo="owner/repo", message=None))

    assert repo.head() == initial_head


def test_submit_includes_precommitted_shared_edits(tmp_path, monkeypatch):
    collection_dir, source, repo, remote = _setup_shared_source_with_remote(tmp_path)
    deck = source.root / "Deck.md"
    deck.write_text(deck.read_text() + "E: precommitted\n", encoding="utf-8")
    repo.commit_paths([deck], "Manual shared edit")
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )
    submitted = []

    def fake_open_pr(_repo, _source, branch, *, title):
        submitted.append((branch, title))
        return True

    monkeypatch.setattr("ankiops.shared.commands.open_pr_if_possible", fake_open_pr)

    run_submit(SimpleNamespace(repo="owner/repo", message=None))

    assert len(submitted) == 1
    assert repo.run(["branch", "--list", submitted[0][0]]).stdout == ""


def test_update_reports_when_shared_source_is_already_current(
    tmp_path,
    monkeypatch,
    caplog,
):
    collection_dir, _source, _repo, remote = _setup_shared_source_with_remote(tmp_path)
    initial_head = _git_head(collection_dir)
    monkeypatch.setattr(
        "ankiops.deck_sources.DeckSource.github_url",
        property(lambda _source: str(remote)),
    )
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with caplog.at_level(logging.INFO, logger="ankiops.shared.commands"):
        run_update(SimpleNamespace(repo="owner/repo"))

    assert _git_head(collection_dir) == initial_head
    assert "owner/repo is already up to date." in caplog.text


@pytest.mark.parametrize("command", ["update", "submit"])
def test_shared_commands_reject_sources_without_subtree_metadata(
    tmp_path,
    monkeypatch,
    command,
):
    collection_dir = _setup_collection(tmp_path)
    shared_root = collection_dir / "shared" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(shared_root / "note_types")
    (shared_root / "Deck.md").write_text(
        "<!-- note_key: key-1 -->\n"
        "<!-- note_type: shared/owner/repo/AnkiOpsQA -->\n"
        "Q: local\nA: deck\n",
        encoding="utf-8",
    )
    _init_git_repo(collection_dir)
    _commit_all(collection_dir)
    monkeypatch.setattr(
        "ankiops.shared.commands.require_collection_dir",
        lambda: collection_dir,
    )

    args = SimpleNamespace(repo="owner/repo", message=None)
    with pytest.raises(ValueError, match="Recreate or re-add"):
        if command == "update":
            run_update(args)
        else:
            run_submit(args)


def test_subtree_commands_use_shared_prefix_and_github_url(tmp_path, monkeypatch):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    calls = []

    def fake_run(repo, args, *, check=True):
        calls.append((repo.collection_dir, args, check))
        stdout = ""
        if args == ["rev-parse", "--verify", "HEAD"]:
            stdout = "head-sha\n"
        elif args[:2] == ["subtree", "split"]:
            stdout = "split-sha\n"
        return subprocess.CompletedProcess(["git", *args], 0, stdout, "")

    monkeypatch.setattr("ankiops.git.CollectionGit.run", fake_run)

    repo = CollectionGit(tmp_path)
    repo.subtree_add(source)
    assert repo.subtree_pull(source) is False
    split_sha = repo.split_subtree(source)
    branch = repo.create_temp_branch(source, split_sha)

    assert branch.startswith("ankiops-shared-owner-repo-")
    commands = [args for _cwd, args, _check in calls]
    assert [
        "subtree",
        "add",
        "--prefix",
        "shared/owner/repo",
        "--message",
        "AnkiOps: add shared/owner/repo from GitHub",
        "https://github.com/owner/repo.git",
        "main",
    ] in commands
    assert [
        "subtree",
        "pull",
        "--prefix",
        "shared/owner/repo",
        "--message",
        "AnkiOps: update shared/owner/repo from GitHub",
        "https://github.com/owner/repo.git",
        "main",
    ] in commands
    assert ["subtree", "split", "--prefix", "shared/owner/repo"] in commands
    assert ["branch", branch, "split-sha"] in commands

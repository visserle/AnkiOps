from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ankiops.collab import _parse_slug, run_contribute, run_publish
from ankiops.collab.hosting import ensure_publish_repo
from ankiops.collab.publish import _unlink_published_source_files
from ankiops.fs import FileSystemAdapter
from ankiops.git import CollectionGit
from ankiops.markdown_format import NOTE_SEPARATOR
from ankiops.sources import SyncSource


def _setup_collection(tmp_path):
    FileSystemAdapter().eject_builtin_note_types(tmp_path / "note_types")
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


def _patch_collection_git(monkeypatch, target="ankiops.collab.commands.CollectionGit"):
    calls = []

    class FakeGit:
        def __init__(self, collection_dir):
            self.collection_dir = collection_dir

        def ensure_repo(self, message):
            calls.append(("ensure_git_repo", self.collection_dir, message))

        def ensure_clean_index(self, message):
            calls.append(("ensure_clean_index", self.collection_dir, message))

        def head(self):
            calls.append(("head", self.collection_dir))
            return "initial-head"

        def commit_publish_move(self, *, touched_paths, source_paths, message):
            calls.append(
                (
                    "commit_publish_move",
                    self.collection_dir,
                    touched_paths,
                    source_paths,
                    message,
                )
            )

        def subtree_split(self, source):
            calls.append(("subtree_split", self.collection_dir, source))
            return "ankiops-test-branch"

        def push_ref(self, url, source_ref, target_ref, *, check=True):
            calls.append(
                ("push_ref", self.collection_dir, url, source_ref, target_ref, check)
            )

        def rollback_to(self, initial_head):
            calls.append(("rollback_to", self.collection_dir, initial_head))

        def delete_branch_if_exists(self, branch):
            calls.append(("delete_branch", self.collection_dir, branch))

        def unstage_or_untrack(self, paths):
            calls.append(("unstage_or_untrack", self.collection_dir, paths))

        def commit_paths(self, paths, message):
            calls.append(("commit_paths", self.collection_dir, paths, message))

        def subtree_add(self, source):
            calls.append(("subtree_add", self.collection_dir, source))

        def subtree_pull(self, source):
            calls.append(("subtree_pull", self.collection_dir, source))

    monkeypatch.setattr(target, FakeGit)
    return calls


def _patch_publish_git(monkeypatch):
    calls = _patch_collection_git(
        monkeypatch,
        "ankiops.collab.publish.CollectionGit",
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda repo, source, *, public: calls.append(
            ("ensure_publish_repo", repo.collection_dir, source, public)
        ),
    )
    return calls


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


def test_publish_unsynced_deck_moves_files_media_and_note_types(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: ![img](media/pic.png)\n",
        encoding="utf-8",
    )
    (collection_dir / "media" / "pic.png").write_bytes(b"img")
    git_calls = _patch_publish_git(monkeypatch)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )

    run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    collab_root = collection_dir / "collab" / "owner" / "repo"
    assert not deck.exists()
    assert (collab_root / "Deck.md").exists()
    content = (collab_root / "Deck.md").read_text(encoding="utf-8")
    assert "<!-- note_type: collab/owner/repo/AnkiOpsQA -->" in content
    assert (collab_root / "media" / "pic.png").read_bytes() == b"img"
    assert (collab_root / "note_types" / "AnkiOpsQA").is_dir()
    assert (collab_root / "note_types" / "AnkiOpsStyling.css").exists()
    assert (collab_root / "note_types" / "SyntaxHighlighting.css").exists()
    FileSystemAdapter().load_note_type_configs(collab_root / "note_types")
    assert any(
        call[0] == "ensure_publish_repo" and call[3] is False for call in git_calls
    )
    assert any(call[0] == "commit_publish_move" for call in git_calls)


def test_publish_missing_repo_blocks_before_anki_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("<!-- note_key: key-1 -->\nQ: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _patch_collection_git(monkeypatch, "ankiops.collab.publish.CollectionGit")
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("GitHub repository does not exist")
        ),
    )

    with pytest.raises(ValueError, match="GitHub repository does not exist"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_publish_rejects_unsafe_repo_slug_before_git_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("Q: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    with pytest.raises(ValueError, match="Invalid GitHub repo slug"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/Segeln_"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_publish_missing_referenced_media_blocks_before_file_mutations(
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
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _patch_collection_git(monkeypatch, "ankiops.collab.publish.CollectionGit")

    with pytest.raises(ValueError, match="referenced media file\\(s\\) missing"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_publish_rejects_missing_note_keys_before_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("Q: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    _patch_collection_git(monkeypatch, "ankiops.collab.publish.CollectionGit")

    with pytest.raises(ValueError, match="missing note_key metadata"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_publish_commit_failure_keeps_source_file(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    subprocess.run(["git", "init"], cwd=collection_dir, check=True, capture_output=True)
    deck = collection_dir / "Deck.md"
    original = "<!-- note_key: key-1 -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.CollectionGit.commit_publish_move",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("commit failed")),
    )

    with pytest.raises(ValueError, match="commit failed"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "collab").exists()


def test_publish_dirty_index_blocks_before_file_mutations_and_preserves_stage(
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
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish._write_publish_files",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )
    monkeypatch.setattr(
        "ankiops.collab.publish._cleanup_failed_publish",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected cleanup")),
    )

    with pytest.raises(ValueError, match="clean git index"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == "M  Other.md\n"
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "collab").exists()


def test_publish_subtree_split_failure_rolls_back_publish_commit(
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
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.CollectionGit.subtree_split",
        lambda *_args: (_ for _ in ()).throw(ValueError("split failed")),
    )

    with pytest.raises(ValueError, match="split failed"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert _git_head(collection_dir) == initial_head
    assert _git_status(collection_dir) == ""
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "collab").exists()


def test_publish_push_failure_rolls_back_publish_commit_and_temp_branch(
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
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.collab.publish.ensure_publish_repo",
        lambda *_args, **_kwargs: None,
    )

    def fake_split(repo, _source):
        subprocess.run(["git", "branch", branch], cwd=repo.collection_dir, check=True)
        return branch

    monkeypatch.setattr(
        "ankiops.collab.publish.CollectionGit.subtree_split", fake_split
    )

    monkeypatch.setattr(
        "ankiops.collab.publish.CollectionGit.push_ref",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("push failed")),
    )

    with pytest.raises(ValueError, match="push failed"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert _git_head(collection_dir) == initial_head
    assert _git_status(collection_dir) == ""
    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "collab").exists()
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
    target = tmp_path / "collab" / "owner" / "repo" / "Segeln.md"
    target.parent.mkdir(parents=True)
    target.write_text("Q: moved\nA: deck\n", encoding="utf-8")

    CollectionGit(tmp_path).commit_paths(
        [target, tmp_path / "Segeln.md"],
        "publish",
    )

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == ""


def test_git_commit_publish_stages_source_removal_before_unlink(tmp_path):
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

    target = tmp_path / "collab" / "owner" / "repo" / "Deck.md"
    target.parent.mkdir(parents=True)
    target.write_text("Q: collab\nA: deck\n", encoding="utf-8")
    plan = SimpleNamespace(files=[SimpleNamespace(source_path=source)])

    CollectionGit(tmp_path).commit_publish_move(
        touched_paths=[target.parent, target, source],
        source_paths=[source],
        message="publish",
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
    assert "A\tcollab/owner/repo/Deck.md" in show.stdout

    _unlink_published_source_files(plan)

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert status.stdout == ""


def test_ensure_publish_repo_without_gh_prints_manual_create_command(
    tmp_path,
    monkeypatch,
):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    monkeypatch.setattr(
        "ankiops.collab.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr("ankiops.collab.hosting.shutil.which", lambda _name: None)

    with pytest.raises(ValueError, match="gh repo create owner/repo --private"):
        ensure_publish_repo(CollectionGit(tmp_path), source, public=False)


def test_ensure_publish_repo_creates_missing_private_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr(
        "ankiops.collab.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr(
        "ankiops.collab.hosting.shutil.which",
        lambda _name: "/bin/gh",
    )
    monkeypatch.setattr(
        "ankiops.collab.hosting._create_github_repo",
        lambda repo, source, *, public: calls.append(
            (repo.collection_dir, source.github_slug, public)
        ),
    )

    ensure_publish_repo(CollectionGit(tmp_path), source, public=False)

    assert calls == [(tmp_path, "owner/repo", False)]


def test_ensure_publish_repo_creates_missing_public_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr(
        "ankiops.collab.hosting._github_repo_exists", lambda *_args: False
    )
    monkeypatch.setattr(
        "ankiops.collab.hosting.shutil.which",
        lambda _name: "/bin/gh",
    )
    monkeypatch.setattr(
        "ankiops.collab.hosting._create_github_repo",
        lambda repo, source, *, public: calls.append(
            (repo.collection_dir, source.github_slug, public)
        ),
    )

    ensure_publish_repo(CollectionGit(tmp_path), source, public=True)

    assert calls == [(tmp_path, "owner/repo", True)]


def test_publish_rejects_duplicate_note_keys(tmp_path, monkeypatch):
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
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )

    with pytest.raises(ValueError, match="Duplicate note_key"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_contribute_rejects_keyless_notes_without_mutating_files(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    collab_root = collection_dir / "collab" / "owner" / "repo"
    FileSystemAdapter().eject_builtin_note_types(collab_root / "note_types")
    deck = collab_root / "Deck.md"
    original = "<!-- note_type: collab/owner/repo/AnkiOpsQA -->\nQ: local\nA: deck\n"
    deck.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        "ankiops.collab.commands.require_collection_dir",
        lambda: collection_dir,
    )
    git_calls = _patch_collection_git(monkeypatch)

    with pytest.raises(ValueError, match="missing note_key metadata"):
        run_contribute(SimpleNamespace(repo="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original
    assert not any(call[0] in {"commit_paths", "subtree_split"} for call in git_calls)


def test_subtree_commands_use_collab_prefix_and_github_url(tmp_path, monkeypatch):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []

    def fake_run(repo, args, *, check=True):
        calls.append((repo.collection_dir, args, check))
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr("ankiops.git.CollectionGit.run", fake_run)

    repo = CollectionGit(tmp_path)
    repo.subtree_add(source)
    repo.subtree_pull(source)
    branch = repo.subtree_split(source)

    assert branch.startswith("ankiops-collab-owner-repo-")
    assert calls[:2] == [
        (
            tmp_path,
            [
                "subtree",
                "add",
                "--prefix",
                "collab/owner/repo",
                "https://github.com/owner/repo.git",
                "main",
            ],
            True,
        ),
        (
            tmp_path,
            [
                "subtree",
                "pull",
                "--prefix",
                "collab/owner/repo",
                "https://github.com/owner/repo.git",
                "main",
            ],
            True,
        ),
    ]
    assert calls[2] == (
        tmp_path,
        [
            "subtree",
            "split",
            "--prefix",
            "collab/owner/repo",
            "-b",
            branch,
        ],
        True,
    )

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ankiops.collab import (
    _ensure_publish_repo,
    _git_commit_paths,
    _git_commit_publish,
    _parse_slug,
    _subtree_add,
    _subtree_pull,
    _subtree_split,
    _unlink_published_source_files,
    run_contribute,
    run_publish,
)
from ankiops.fs import FileSystemAdapter
from ankiops.markdown_format import NOTE_SEPARATOR
from ankiops.sources import SyncSource


def _setup_collection(tmp_path):
    FileSystemAdapter().eject_builtin_note_types(tmp_path / "note_types")
    (tmp_path / "media").mkdir()
    return tmp_path


def _patch_publish_git(monkeypatch):
    calls = []

    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)
    monkeypatch.setattr(
        "ankiops.collab._ensure_publish_repo",
        lambda collection_dir, source, *, public: calls.append(
            ("ensure_repo", collection_dir, source, public)
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab._git_commit_publish",
        lambda collection_dir, plan, paths, message: calls.append(
            ("commit", collection_dir, plan, paths, message)
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab._subtree_split",
        lambda collection_dir, source: "ankiops-test-branch",
    )

    def fake_run_git(collection_dir, args):
        calls.append(("git", collection_dir, args))

    monkeypatch.setattr("ankiops.collab._run_git", fake_run_git)
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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)

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
    assert any(call[0] == "ensure_repo" and call[3] is False for call in git_calls)
    assert any(call[0] == "commit" for call in git_calls)


def test_publish_missing_repo_blocks_before_anki_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("<!-- note_key: key-1 -->\nQ: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)
    monkeypatch.setattr(
        "ankiops.collab._ensure_publish_repo",
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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.collab._ensure_git_repo",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected git check")),
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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)

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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)

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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)
    monkeypatch.setattr(
        "ankiops.collab._ensure_publish_repo",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ankiops.collab._git_commit_publish",
        lambda *_args: (_ for _ in ()).throw(ValueError("commit failed")),
    )

    with pytest.raises(ValueError, match="commit failed"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original
    assert not (collection_dir / "collab").exists()


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

    _git_commit_paths(tmp_path, [target, tmp_path / "Segeln.md"], "publish")

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

    _git_commit_publish(
        tmp_path,
        plan,
        [target.parent, target, source],
        "publish",
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
    monkeypatch.setattr("ankiops.collab._github_repo_exists", lambda *_args: False)
    monkeypatch.setattr("ankiops.collab.shutil.which", lambda _name: None)

    with pytest.raises(ValueError, match="gh repo create owner/repo --private"):
        _ensure_publish_repo(tmp_path, source, public=False)


def test_ensure_publish_repo_creates_missing_private_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr("ankiops.collab._github_repo_exists", lambda *_args: False)
    monkeypatch.setattr("ankiops.collab.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        "ankiops.collab._create_github_repo",
        lambda collection_dir, source, *, public: calls.append(
            (collection_dir, source.github_slug, public)
        ),
    )

    _ensure_publish_repo(tmp_path, source, public=False)

    assert calls == [(tmp_path, "owner/repo", False)]


def test_ensure_publish_repo_creates_missing_public_repo_with_gh(
    tmp_path,
    monkeypatch,
):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []
    monkeypatch.setattr("ankiops.collab._github_repo_exists", lambda *_args: False)
    monkeypatch.setattr("ankiops.collab.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        "ankiops.collab._create_github_repo",
        lambda collection_dir, source, *, public: calls.append(
            (collection_dir, source.github_slug, public)
        ),
    )

    _ensure_publish_repo(tmp_path, source, public=True)

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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)

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
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)
    monkeypatch.setattr(
        "ankiops.collab._git_commit_paths",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected commit")),
    )
    monkeypatch.setattr(
        "ankiops.collab._subtree_split",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected split")),
    )

    with pytest.raises(ValueError, match="missing note_key metadata"):
        run_contribute(SimpleNamespace(repo="owner/repo"))

    assert deck.read_text(encoding="utf-8") == original


def test_subtree_commands_use_collab_prefix_and_github_url(tmp_path, monkeypatch):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    calls = []

    def fake_run_git(collection_dir, args):
        calls.append((collection_dir, args))

    monkeypatch.setattr("ankiops.collab._run_git", fake_run_git)

    _subtree_add(tmp_path, source)
    _subtree_pull(tmp_path, source)
    branch = _subtree_split(tmp_path, source)

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
    )

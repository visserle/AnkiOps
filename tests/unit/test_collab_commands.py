from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ankiops.ankiops_bridge import AnkiOpsBridgeError
from ankiops.collab import (
    _ensure_publish_repo,
    _git_commit_paths,
    _parse_slug,
    _subtree_add,
    _subtree_pull,
    _subtree_split,
    run_contribute,
    run_publish,
)
from ankiops.fs import FileSystemAdapter
from ankiops.markdown_format import NOTE_SEPARATOR
from ankiops.models import AnkiNote
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
        "ankiops.collab._git_commit_paths",
        lambda collection_dir, paths, message: calls.append(
            ("commit", collection_dir, paths, message)
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


def _empty_anki():
    anki = MagicMock()
    anki.fetch_model_names.return_value = []
    anki.fetch_all_note_ids.return_value = []
    anki.fetch_notes_info.return_value = {}
    anki.fetch_cards_info.return_value = {}
    return anki


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
    deck.write_text("Q: local\nA: ![img](media/pic.png)\n", encoding="utf-8")
    (collection_dir / "media" / "pic.png").write_bytes(b"img")
    anki = _empty_anki()
    git_calls = _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

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
    anki.change_notes_notetype.assert_not_called()
    assert any(call[0] == "ensure_repo" and call[3] is False for call in git_calls)
    assert any(call[0] == "commit" for call in git_calls)


def test_publish_missing_repo_blocks_before_anki_or_file_mutations(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text("Q: local\nA: deck\n", encoding="utf-8")
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)
    monkeypatch.setattr(
        "ankiops.collab._ensure_publish_repo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("GitHub repository does not exist")
        ),
    )
    monkeypatch.setattr(
        "ankiops.collab.connect_or_exit",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Anki connect")),
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
    monkeypatch.setattr(
        "ankiops.collab.connect_or_exit",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Anki connect")),
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
    deck.write_text("Q: local\nA: ![img](media/missing.png)\n", encoding="utf-8")
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab._ensure_git_repo", lambda _collection_dir: None)

    with pytest.raises(ValueError, match="referenced media file\\(s\\) missing"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
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


def test_publish_synced_deck_creates_scoped_model_and_calls_bridge(
    tmp_path, monkeypatch
):
    collection_dir = _setup_collection(tmp_path)
    (collection_dir / "Deck.md").write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: deck\n",
        encoding="utf-8",
    )
    anki = MagicMock()
    anki.fetch_all_note_ids.return_value = [100]
    anki.fetch_notes_info.return_value = {
        100: AnkiNote(
            note_id=100,
            note_type="AnkiOpsQA",
            fields={"AnkiOps Key": "key-1"},
            card_ids=[1000],
        )
    }
    anki.fetch_cards_info.return_value = {
        1000: {"cardId": 1000, "note": 100, "deckName": "Deck"}
    }
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

    run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    created = anki.create_models.call_args.args[0]
    assert [config.name for config in created] == ["collab/owner/repo/AnkiOpsQA"]
    anki.change_notes_notetype.assert_called_once_with(
        [100],
        "AnkiOpsQA",
        "collab/owner/repo/AnkiOpsQA",
    )


def test_publish_shared_model_converts_only_selected_deck_notes(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    (collection_dir / "Deck.md").write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: deck\n",
        encoding="utf-8",
    )
    anki = MagicMock()
    anki.fetch_all_note_ids.return_value = [100, 101]
    anki.fetch_notes_info.return_value = {
        100: AnkiNote(
            note_id=100,
            note_type="AnkiOpsQA",
            fields={"AnkiOps Key": "key-1"},
            card_ids=[1000],
        ),
        101: AnkiNote(
            note_id=101,
            note_type="AnkiOpsQA",
            fields={"AnkiOps Key": "other-key"},
            card_ids=[1001],
        ),
    }
    anki.fetch_cards_info.return_value = {
        1000: {"cardId": 1000, "note": 100, "deckName": "Deck"},
        1001: {"cardId": 1001, "note": 101, "deckName": "Other"},
    }
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

    run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    anki.change_notes_notetype.assert_called_once_with(
        [100],
        "AnkiOpsQA",
        "collab/owner/repo/AnkiOpsQA",
    )


def test_publish_blocks_note_already_scoped_to_different_collab_slug(
    tmp_path,
    monkeypatch,
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: deck\n",
        encoding="utf-8",
    )
    anki = MagicMock()
    anki.fetch_model_names.return_value = [
        "AnkiOpsQA",
        "collab/owner/old/AnkiOpsQA",
    ]
    anki.fetch_all_note_ids.return_value = [100]
    anki.fetch_notes_info.return_value = {
        100: AnkiNote(
            note_id=100,
            note_type="collab/owner/old/AnkiOpsQA",
            fields={"AnkiOps Key": "key-1"},
            card_ids=[1000],
        )
    }
    anki.fetch_cards_info.return_value = {
        1000: {"cardId": 1000, "note": 100, "deckName": "Deck"}
    }
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

    with pytest.raises(ValueError, match="same owner/repo slug"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()
    anki.change_notes_notetype.assert_not_called()


def test_publish_blocks_note_with_cards_inside_and_outside_scope(
    tmp_path, monkeypatch
):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: deck\n",
        encoding="utf-8",
    )
    anki = MagicMock()
    anki.fetch_all_note_ids.return_value = [100]
    anki.fetch_notes_info.return_value = {
        100: AnkiNote(
            note_id=100,
            note_type="AnkiOpsQA",
            fields={"AnkiOps Key": "key-1"},
            card_ids=[1000, 1001],
        )
    }
    anki.fetch_cards_info.return_value = {
        1000: {"cardId": 1000, "note": 100, "deckName": "Deck"},
        1001: {"cardId": 1001, "note": 100, "deckName": "Other"},
    }
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

    with pytest.raises(ValueError, match="both inside"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


def test_publish_bridge_failure_leaves_files_untouched(tmp_path, monkeypatch):
    collection_dir = _setup_collection(tmp_path)
    deck = collection_dir / "Deck.md"
    deck.write_text(
        "<!-- note_key: key-1 -->\nQ: local\nA: deck\n",
        encoding="utf-8",
    )
    anki = MagicMock()
    anki.fetch_all_note_ids.return_value = [100]
    anki.fetch_notes_info.return_value = {
        100: AnkiNote(
            note_id=100,
            note_type="AnkiOpsQA",
            fields={"AnkiOps Key": "key-1"},
            card_ids=[1000],
        )
    }
    anki.fetch_cards_info.return_value = {
        1000: {"cardId": 1000, "note": 100, "deckName": "Deck"}
    }
    anki.fetch_model_names.return_value = ["AnkiOpsQA"]
    anki.change_notes_notetype.side_effect = AnkiOpsBridgeError("bridge down")
    _patch_publish_git(monkeypatch)
    monkeypatch.setattr("ankiops.collab.require_collection_dir", lambda: collection_dir)
    monkeypatch.setattr("ankiops.collab.connect_or_exit", lambda: anki)

    with pytest.raises(ValueError, match="bridge down"):
        run_publish(SimpleNamespace(deck="Deck", repo="owner/repo"))

    assert deck.exists()
    assert not (collection_dir / "collab").exists()


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

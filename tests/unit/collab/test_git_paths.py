from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankiops.collab.commands import (
    INTEGRATED_REF,
    _delete_submission_branch,
    _derive_submit_title,
    _github_slug_from_remote,
    _parse_github_slug,
    _tree_delta_is_present,
)
from ankiops.deck_sources import DeckSource
from ankiops.git import GitRepository


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _init_repository(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Path Test")
    _git(root, "config", "user.email", "path@example.test")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def test_unicode_deck_rename_has_human_submit_title(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    (tmp_path / "Deck.md").write_text("content\n", encoding="utf-8")
    base = _commit(tmp_path, "root")
    _git(tmp_path, "mv", "Deck.md", "Déck Ω — spaced.md")
    _commit(tmp_path, "rename")

    assert _derive_submit_title(GitRepository(tmp_path), base) == (
        "Rename Deck to Déck Ω — spaced"
    )


def test_unrelated_candidate_does_not_contain_unicode_path_change(
    tmp_path: Path,
) -> None:
    _init_repository(tmp_path)
    deck = tmp_path / "Déck Ω — spaced.md"
    deck.write_text("before\n", encoding="utf-8")
    base = _commit(tmp_path, "root")

    deck.write_text("uploaded\n", encoding="utf-8")
    changed = _commit(tmp_path, "uploaded Unicode change")

    _git(tmp_path, "checkout", "-b", "candidate", base)
    (tmp_path / "README.md").write_text("unrelated\n", encoding="utf-8")
    candidate = _commit(tmp_path, "unrelated upstream change")

    assert not _tree_delta_is_present(
        GitRepository(tmp_path),
        base_tree=base,
        changed_tree=changed,
        candidate_tree=candidate,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/Owner/Repo.git",
        "git@github.com:Owner/Repo.git",
        "ssh://git@GitHub.com/Owner/Repo.git",
    ],
)
def test_github_slug_accepts_standard_remote_url_forms(url: str) -> None:
    assert _github_slug_from_remote(url) == "Owner/Repo"


@pytest.mark.parametrize(
    "slug",
    [
        "visserle/QA_",
        "managed_user/QA",
        "owner/QA.v2_",
        "owner/_",
        "owner/-",
        "owner/.gitignore",
    ],
)
def test_collab_identity_accepts_github_repository_names(slug: str) -> None:
    assert _parse_github_slug(slug) == slug


@pytest.mark.parametrize("slug", ["owner/.", "owner/.."])
def test_collab_identity_rejects_path_segments(slug: str) -> None:
    with pytest.raises(ValueError, match="Invalid collab deck identity"):
        _parse_github_slug(slug)


def test_merged_cleanup_keeps_a_branch_advanced_after_merge(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    source = DeckSource.collab(collection, "owner/repo")
    source.root.mkdir(parents=True)
    _init_repository(source.root)
    deck = source.root / "Deck.md"
    deck.write_text("base\n", encoding="utf-8")
    base = _commit(source.root, "Base")
    repository = GitRepository(source.root)
    repository.update_ref(INTEGRATED_REF, base)

    publish = tmp_path / "publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _git(source.root, "remote", "add", "publish", str(publish))
    _git(source.root, "checkout", "-b", "ankiops/contribution")
    deck.write_text("submitted\n", encoding="utf-8")
    submitted = _commit(source.root, "Submit")
    _git(source.root, "push", "publish", "HEAD:ankiops/contribution")
    deck.write_text("unrelated post-merge work\n", encoding="utf-8")
    reused_head = _commit(source.root, "Reuse branch after merge")
    _git(source.root, "push", "publish", "HEAD:ankiops/contribution")

    deleted = _delete_submission_branch(
        source,
        repository,
        {
            "publish_branch": "ankiops/contribution",
            "pushed_sha": submitted,
        },
        SimpleNamespace(
            state="MERGED",
            head_branch="ankiops/contribution",
            head_sha=reused_head,
            head_owner="",
            head_repository="",
        ),
    )

    assert not deleted
    assert (
        repository.remote_branch_sha("publish", "ankiops/contribution") == reused_head
    )

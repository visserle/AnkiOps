from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ankiops.collab.commands import (
    INTEGRATED_REF,
    _delete_submission_branch,
    _derive_submit_title,
    _github_slug_from_remote,
    _SubmissionPhase,
    _SubmissionState,
)
from ankiops.deck_sources import parse_github_slug
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
    assert parse_github_slug(slug) == slug


@pytest.mark.parametrize("slug", ["owner/.", "owner/.."])
def test_collab_identity_rejects_path_segments(slug: str) -> None:
    with pytest.raises(ValueError, match="Invalid collab deck identity"):
        parse_github_slug(slug)


def test_merged_cleanup_keeps_a_branch_advanced_after_merge(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    source = collection / "collab" / "owner" / "repo"
    source.mkdir(parents=True)
    _init_repository(source)
    deck = source / "Deck.md"
    deck.write_text("base\n", encoding="utf-8")
    base = _commit(source, "Base")
    repository = GitRepository(source)
    repository.update_ref(INTEGRATED_REF, base)

    publish = tmp_path / "publish.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(publish))
    _git(source, "remote", "add", "publish", str(publish))
    _git(source, "checkout", "-b", "ankiops/contribution")
    deck.write_text("submitted\n", encoding="utf-8")
    submitted = _commit(source, "Submit")
    _git(source, "push", "publish", "HEAD:ankiops/contribution")
    deck.write_text("unrelated post-merge work\n", encoding="utf-8")
    reused_head = _commit(source, "Reuse branch after merge")
    _git(source, "push", "publish", "HEAD:ankiops/contribution")

    deleted = _delete_submission_branch(
        repository,
        _SubmissionState(
            phase=_SubmissionPhase.MERGED,
            snapshot=submitted,
            uploaded=submitted,
            publish_slug=None,
            remote_sha=reused_head,
            pull_request=None,
        ),
    )

    assert not deleted
    assert (
        repository.remote_branch_sha("publish", "ankiops/contribution") == reused_head
    )

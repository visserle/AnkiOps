"""GitHub hosting operations for shared sources."""

from __future__ import annotations

import logging
import shutil
import subprocess

from ankiops.deck_sources import SHARED_BRANCH, DeckSource
from ankiops.git import CollectionGit

logger = logging.getLogger(__name__)


def _visibility_flag(public: bool) -> str:
    return "--public" if public else "--private"


def _manual_repo_create_command(source: DeckSource, *, public: bool) -> str:
    slug = source.github_slug or source.source_id
    return f"gh repo create {slug} {_visibility_flag(public)}"


def _github_repo_exists(repo: CollectionGit, source: DeckSource) -> bool:
    slug = source.github_slug
    if slug is None or source.github_url is None:
        raise ValueError(f"Cannot derive GitHub repo for {source.display_name}")

    gh_path = shutil.which("gh")
    if gh_path:
        result = subprocess.run(
            [gh_path, "repo", "view", slug],
            cwd=repo.collection_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    return repo.remote_exists(source.github_url)


def _create_github_repo(
    repo: CollectionGit,
    source: DeckSource,
    *,
    public: bool,
) -> None:
    slug = source.github_slug
    gh_path = shutil.which("gh")
    if slug is None or gh_path is None:
        raise ValueError(
            "GitHub repository does not exist or is not accessible. "
            "Create it first with: "
            f"{_manual_repo_create_command(source, public=public)}"
        )

    logger.info("Creating GitHub repository %s", slug)
    result = subprocess.run(
        [gh_path, "repo", "create", slug, _visibility_flag(public)],
        cwd=repo.collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise ValueError(
            f"GitHub repository {slug} does not exist and AnkiOps could not "
            f"create it with gh: {detail}. Create it first with: "
            f"{_manual_repo_create_command(source, public=public)}"
        )


def ensure_create_repo(
    repo: CollectionGit,
    source: DeckSource,
    *,
    public: bool,
) -> None:
    if _github_repo_exists(repo, source):
        return
    if shutil.which("gh") is None:
        raise ValueError(
            "GitHub repository does not exist or is not accessible: "
            f"{source.github_slug}. Create it first with: "
            f"{_manual_repo_create_command(source, public=public)}"
        )
    _create_github_repo(repo, source, public=public)


def open_pr_if_possible(
    repo: CollectionGit,
    source: DeckSource,
    branch: str,
    *,
    title: str,
) -> bool:
    if source.github_url is None:
        logger.info("Prepared branch %s. Push it and open a PR manually.", branch)
        return False
    push = repo.push_ref(source.github_url, branch, branch, check=False)
    if push.returncode != 0:
        logger.info(
            "Prepared branch %s. Push it to %s and open a PR manually.",
            branch,
            source.github_url,
        )
        logger.debug(push.stderr)
        return False

    if shutil.which("gh") is None:
        logger.info(
            "Pushed branch %s to %s. Open a PR manually.",
            branch,
            source.github_url,
        )
        return True

    gh = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            source.github_slug or "",
            "--head",
            branch,
            "--base",
            SHARED_BRANCH,
            "--title",
            title,
            "--body",
            f"Submitted by AnkiOps from {source.source_id}.",
        ],
        cwd=repo.collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if gh.returncode == 0:
        logger.info(gh.stdout.strip())
    else:
        logger.info("Pushed branch %s. Open the PR manually.", branch)
        logger.debug(gh.stderr)
    return True

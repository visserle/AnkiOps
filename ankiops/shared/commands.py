"""GitHub-native shared source commands."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rich.markup import escape as rich_escape

from ankiops.collection import require_collection_dir
from ankiops.console import clickable_path
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
)
from ankiops.git import CollectionGit
from ankiops.markdown import read_deck_file
from ankiops.shared.create import create_shared_deck
from ankiops.shared.errors import format_missing_note_keys_error
from ankiops.shared.hosting import open_pr_if_possible

logger = logging.getLogger(__name__)

_SAFE_SLUG_PART_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
_GITHUB_OWNER_MAX_LENGTH = 39
_GITHUB_REPO_MAX_LENGTH = 100


def _parse_slug(slug: str) -> tuple[str, str]:
    parts = slug.strip().removesuffix(".git").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Expected GitHub repo as owner/repo")
    owner, repo = parts
    _validate_slug_part(
        slug=slug,
        label="owner",
        value=owner,
        max_length=_GITHUB_OWNER_MAX_LENGTH,
    )
    _validate_slug_part(
        slug=slug,
        label="repo",
        value=repo,
        max_length=_GITHUB_REPO_MAX_LENGTH,
    )
    return owner, repo


def _validate_slug_part(
    *,
    slug: str,
    label: str,
    value: str,
    max_length: int,
) -> None:
    if len(value) > max_length or not _SAFE_SLUG_PART_RE.fullmatch(value):
        raise ValueError(
            f"Invalid GitHub repo slug '{slug}'. AnkiOps shared {label} names "
            "must use only ASCII letters, digits, and hyphens, must start and "
            "end with a letter or digit, and must be "
            f"{max_length} characters or fewer."
        )


def _source_for_slug(collection_dir: Path, slug: str) -> DeckSource:
    owner, repo = _parse_slug(slug)
    return DeckSource.shared(collection_dir, owner, repo)


def _ensure_submittable_note_keys(source: DeckSource) -> None:
    configs = load_note_types_for_source(source)
    missing: list[str] = []

    for md_file in source.deck_files():
        parsed = read_deck_file(md_file, note_types=configs, context_root=source.root)
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing.append(f"{md_file.relative_to(source.root)} note {index}")

    if missing:
        raise ValueError(format_missing_note_keys_error(len(missing)))


def _ensure_compatible_history(repo: CollectionGit, source: DeckSource) -> None:
    if not repo.has_subtree_metadata(source):
        raise ValueError(
            f"Shared source {source.source_id} has incompatible git history. "
            "Recreate or re-add it with this AnkiOps version."
        )


def _remote_state(repo: CollectionGit, source: DeckSource) -> tuple[str, str, str]:
    remote_head = repo.fetch_source_head(source)
    split_sha = repo.split_subtree(source)
    if split_sha == remote_head:
        return remote_head, split_sha, "current"
    if repo.is_ancestor(split_sha, remote_head):
        return remote_head, split_sha, "remote_ahead"
    if repo.trees_equal(split_sha, remote_head):
        return remote_head, split_sha, "same_tree"
    if repo.is_ancestor(remote_head, split_sha):
        return remote_head, split_sha, "local_ahead"
    if repo.has_common_ancestor(split_sha, remote_head):
        return remote_head, split_sha, "diverged"
    return remote_head, split_sha, "incompatible"


def run_create(args) -> None:
    collection_dir = require_collection_dir()
    source = _source_for_slug(collection_dir, args.repo)
    create_shared_deck(
        collection_dir,
        args.deck,
        source,
        public=bool(getattr(args, "public", False)),
    )
    logger.info(
        "Created shared source %s from %s. "
        "Run 'ankiops fa' to apply scoped note types to Anki.",
        source.source_id,
        args.deck,
    )


def run_add(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    if source.root.exists():
        raise ValueError(f"Shared source already exists: {source.source_id}")
    repo.subtree_add(source)
    logger.info(
        "Added %s at %s",
        rich_escape(args.repo),
        clickable_path(source.root),
        extra={"markup": True},
    )


def run_update(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    requested_source = None
    if args.repo:
        requested_source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    sources = discover_deck_sources(collection_dir)
    if requested_source is not None:
        if not requested_source.root.exists():
            raise ValueError(f"Unknown shared source: {requested_source.source_id}")
        targets = [requested_source]
    else:
        targets = [source for source in sources if source.is_shared]
    if not targets:
        logger.info("No shared sources found.")
        return
    for source in targets:
        _ensure_compatible_history(repo, source)
    for source in targets:
        if repo.subtree_pull(source):
            logger.info("Updated %s", source.source_id)
        else:
            logger.info(
                "%s is already up to date.",
                source.github_slug or source.source_id,
            )


def run_submit(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    if not source.root.exists():
        raise ValueError(f"Unknown shared source: {source.source_id}")
    _ensure_submittable_note_keys(source)
    _ensure_compatible_history(repo, source)
    title = getattr(args, "message", None) or f"Update shared deck {args.repo}"
    shared_changes = repo.status_lines([repo.source_prefix(source)])
    if shared_changes and not getattr(args, "commit", False):
        raise ValueError(
            f"Shared source {args.repo} has uncommitted changes. Review them with "
            f"'ankiops shared status {args.repo}', commit them yourself, or re-run "
            "with '--commit'."
        )
    if shared_changes:
        repo.commit_paths(
            [source.root],
            f"AnkiOps: {title}",
        )
    _remote_head, split_sha, remote_state = _remote_state(repo, source)
    if remote_state in {"current", "remote_ahead", "same_tree"}:
        logger.info("No shared changes to submit for %s.", args.repo)
        return
    if remote_state == "incompatible":
        raise ValueError(
            f"Shared source {source.source_id} has incompatible git history. "
            "Recreate or re-add it with this AnkiOps version."
        )
    repo.rejoin_subtree(
        source,
        split_sha,
        f"AnkiOps: record submission history for {source.source_id}",
    )
    branch = repo.create_temp_branch(source, split_sha)
    pushed = open_pr_if_possible(repo, source, branch, title=title)
    if pushed:
        repo.delete_branch_if_exists(branch)


def run_status(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    if not source.root.exists():
        raise ValueError(f"Unknown shared source: {source.source_id}")
    _ensure_compatible_history(repo, source)

    prefix = repo.source_prefix(source)
    shared_changes = repo.status_lines([prefix])
    private_changes = repo.status_lines(
        [".", f":(exclude){prefix}", f":(exclude){prefix}/**"]
    )
    _remote_head, _split_sha, remote_state = _remote_state(repo, source)

    logger.info("Shared source: %s", args.repo)
    _log_status_changes("Shared changes", shared_changes)
    _log_status_changes("Private changes", private_changes, preserved=True)
    logger.info("Remote: %s", _remote_status_message(remote_state))
    logger.info(
        "Submit: %s",
        _submit_status_message(args.repo, shared_changes, remote_state),
    )


def _log_status_changes(
    label: str,
    changes: list[str],
    *,
    preserved: bool = False,
) -> None:
    suffix = " (preserved)" if preserved and changes else ""
    logger.info("%s: %d%s", label, len(changes), suffix)
    for change in changes:
        logger.info("  %s", change)


def _remote_status_message(state: str) -> str:
    return {
        "current": "Committed shared state matches GitHub main.",
        "remote_ahead": "GitHub main is ahead of the committed shared state.",
        "same_tree": "Committed shared content matches GitHub main; history differs.",
        "local_ahead": "Committed shared state has changes not on GitHub main.",
        "diverged": "Committed shared state and GitHub main have both advanced.",
        "incompatible": (
            "Committed shared history has no common ancestor with GitHub main."
        ),
    }[state]


def _submit_status_message(
    slug: str,
    shared_changes: list[str],
    remote_state: str,
) -> str:
    if shared_changes:
        return (
            "blocked by uncommitted shared files; commit them yourself or run "
            f"'ankiops shared submit {slug} --commit'."
        )
    if remote_state in {"current", "same_tree"}:
        return "no-op; there are no shared changes to submit."
    if remote_state == "remote_ahead":
        return f"no-op; run 'ankiops shared update {slug}' to receive GitHub changes."
    if remote_state in {"local_ahead", "diverged"}:
        return "will open a pull request."
    return "blocked by incompatible history; recreate or re-add this source."


def run_list(args) -> None:
    collection_dir = require_collection_dir()
    sources = [
        source for source in discover_deck_sources(collection_dir) if source.is_shared
    ]
    if not sources:
        logger.info("No shared sources found.")
        return
    for source in sources:
        logger.info(
            "%s  %s",
            rich_escape(source.source_id),
            clickable_path(source.root),
            extra={"markup": True},
        )


def run(args) -> None:
    match args.shared_command:
        case "create":
            run_create(args)
        case "add":
            run_add(args)
        case "update":
            run_update(args)
        case "submit":
            run_submit(args)
        case "status":
            run_status(args)
        case "list":
            run_list(args)
        case _:
            raise SystemExit(2)

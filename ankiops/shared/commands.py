"""GitHub-native shared source commands."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rich.markup import escape as rich_escape

from ankiops.config import require_collection_dir
from ankiops.fs import FileSystemAdapter
from ankiops.git import CollectionGit
from ankiops.log import clickable_path
from ankiops.shared.create import create_shared_deck
from ankiops.shared.hosting import open_pr_if_possible
from ankiops.sources import (
    SyncSource,
    discover_sync_sources,
    load_configs_for_source,
    markdown_files_for_source,
)

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


def _source_for_slug(collection_dir: Path, slug: str) -> SyncSource:
    owner, repo = _parse_slug(slug)
    return SyncSource.shared(collection_dir, owner, repo)


def _ensure_submittable_note_keys(source: SyncSource) -> None:
    configs = load_configs_for_source(source)
    parser = FileSystemAdapter()
    parser.set_configs(configs)
    missing: list[str] = []

    for md_file in markdown_files_for_source(source):
        parsed = parser.read_markdown_file(md_file, context_root=source.root)
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing.append(f"{md_file.relative_to(source.root)} note {index}")

    if missing:
        raise ValueError(
            "Cannot submit: notes are missing note_key metadata: "
            + ", ".join(missing)
            + ". Run 'ankiops ma' first or add explicit note_key comments."
        )


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
        "Run 'ankiops ma' to apply scoped note types to Anki.",
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
    sources = discover_sync_sources(collection_dir)
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
        repo.subtree_pull(source)
        logger.info("Updated %s", source.source_id)


def run_submit(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    if not source.root.exists():
        raise ValueError(f"Unknown shared source: {source.source_id}")
    _ensure_submittable_note_keys(source)
    repo.commit_paths(
        [source.root],
        f"AnkiOps: submit {source.source_id}",
    )
    branch = repo.subtree_split(source)
    open_pr_if_possible(repo, source, branch)


def run_list(args) -> None:
    collection_dir = require_collection_dir()
    sources = [
        source for source in discover_sync_sources(collection_dir) if source.is_shared
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
        case "list":
            run_list(args)
        case _:
            raise SystemExit(2)

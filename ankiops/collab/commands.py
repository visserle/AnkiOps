"""GitHub-native collaboration source commands."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ankiops.collab.hosting import open_pr_if_possible
from ankiops.collab.publish import publish_deck
from ankiops.config import require_collection_dir
from ankiops.fs import FileSystemAdapter
from ankiops.git import CollectionGit
from ankiops.log import clickable_path
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
            f"Invalid GitHub repo slug '{slug}'. AnkiOps collab {label} names "
            "must use only ASCII letters, digits, and hyphens, must start and "
            "end with a letter or digit, and must be "
            f"{max_length} characters or fewer."
        )


def _source_for_slug(collection_dir: Path, slug: str) -> SyncSource:
    owner, repo = _parse_slug(slug)
    return SyncSource.collab(collection_dir, owner, repo)


def _ensure_contributable_note_keys(source: SyncSource) -> None:
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
            "Cannot contribute: notes are missing note_key metadata: "
            + ", ".join(missing)
            + ". Run 'ankiops ma' first or add explicit note_key comments."
        )


def run_publish(args) -> None:
    collection_dir = require_collection_dir()
    source = _source_for_slug(collection_dir, args.repo)
    publish_deck(
        collection_dir,
        args.deck,
        source,
        public=bool(getattr(args, "public", False)),
    )
    logger.info(
        "Published %s to %s. Run 'ankiops ma' to apply scoped note types to Anki.",
        args.deck,
        source.source_id,
    )


def run_subscribe(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Collab commands require a git-backed collection.")
    if source.root.exists():
        raise ValueError(f"Collab source already exists: {source.source_id}")
    repo.subtree_add(source)
    logger.info("Subscribed to %s at %s", args.repo, clickable_path(source.root))


def run_pull(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    requested_source = None
    if args.repo:
        requested_source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Collab commands require a git-backed collection.")
    sources = discover_sync_sources(collection_dir)
    if requested_source is not None:
        if not requested_source.root.exists():
            raise ValueError(f"Unknown collab source: {requested_source.source_id}")
        targets = [requested_source]
    else:
        targets = [source for source in sources if source.is_collab]
    if not targets:
        logger.info("No collab sources found.")
        return
    for source in targets:
        repo.subtree_pull(source)
        logger.info("Pulled %s", source.source_id)


def run_contribute(args) -> None:
    collection_dir = require_collection_dir()
    repo = CollectionGit(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo.ensure_repo("Collab commands require a git-backed collection.")
    if not source.root.exists():
        raise ValueError(f"Unknown collab source: {source.source_id}")
    _ensure_contributable_note_keys(source)
    repo.commit_paths(
        [source.root],
        f"AnkiOps: contribute {source.source_id}",
    )
    branch = repo.subtree_split(source)
    open_pr_if_possible(repo, source, branch)


def run_status(args) -> None:
    collection_dir = require_collection_dir()
    sources = [
        source for source in discover_sync_sources(collection_dir) if source.is_collab
    ]
    if not sources:
        logger.info("No collab sources found.")
        return
    for source in sources:
        logger.info("%s  %s", source.source_id, clickable_path(source.root))


def run(args) -> None:
    match args.collab_command:
        case "publish":
            run_publish(args)
        case "subscribe":
            run_subscribe(args)
        case "pull":
            run_pull(args)
        case "contribute":
            run_contribute(args)
        case "status":
            run_status(args)
        case _:
            raise SystemExit(2)

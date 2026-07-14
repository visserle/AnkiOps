"""Publish local decks as GitHub collab sources."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

from ankiops.collab.errors import RepositoryCollisionError
from ankiops.collab.git_state import (
    INTEGRATED_REF,
    JOURNAL_BRANCH,
    PUBLISH_DECK_CONFIG,
    PUBLISH_PREPARED_REF,
)
from ankiops.collab.hosting import GitHubHost
from ankiops.collection import LOCAL_MEDIA_DIR, NOTE_TYPES_DIR, file_stem_to_deck_name
from ankiops.deck_sources import DeckSource
from ankiops.git import GitRepository
from ankiops.interchange import ParsedDeck, parse_source, require_note_keys
from ankiops.markdown import render_notes_to_markdown
from ankiops.media import extract_media_references
from ankiops.note_types import NoteType
from ankiops.sync.state import SyncState


@dataclass(frozen=True)
class _RenderedPublishFile:
    source_path: Path
    target_path: Path
    content: str


@dataclass(frozen=True)
class _PublishAsset:
    relative_path: Path
    content: bytes


@dataclass(frozen=True)
class _PublishPlan:
    deck: str
    source: DeckSource
    files: list[_RenderedPublishFile]
    note_keys: set[str]
    assets: tuple[_PublishAsset, ...]


@dataclass(frozen=True)
class PreparedPublish:
    prepared_commit: str
    deck: str


def prepared_publish(repository: GitRepository) -> PreparedPublish | None:
    """Read the durable state of an interrupted publish."""
    prepared_commit = repository.ref_sha(PUBLISH_PREPARED_REF)
    deck = repository.config_get(PUBLISH_DECK_CONFIG)
    if prepared_commit is None and deck is None:
        return None
    if deck is None:
        raise ValueError(
            f"The prepared publish state at {repository.root} is incomplete."
        )
    if prepared_commit is None:
        prepared_commit = repository.head()
    if prepared_commit is None:
        raise ValueError(
            f"The prepared publish state at {repository.root} has no commit."
        )
    return PreparedPublish(prepared_commit, deck)


def publish_collab_deck(
    collection_root: Path,
    deck: str,
    source: DeckSource,
) -> None:
    collection_git = GitRepository(collection_root)
    collection_git.ensure_repo(
        "AnkiOps collections require a Git repository at the root."
    )
    source_git = GitRepository(source.root)
    prepared_head: str | None
    if source.root.exists():
        if not source_git.is_repo():
            raise RepositoryCollisionError(
                f"Local collab source already exists and is not a Git repository: "
                f"{source.root}"
            )
        preparation = prepared_publish(source_git)
        if preparation is None:
            raise RepositoryCollisionError(
                f"Local collab source already exists: {source.display_name}. "
                "It was not prepared by this publish, so it was left untouched."
            )
        if preparation.deck != deck:
            raise ValueError(
                f"The interrupted publish at {source.root} is for "
                f"'{preparation.deck}', "
                f"not '{deck}'. Retry with that deck name."
            )
        prepared_marker = source_git.ref_sha(PUBLISH_PREPARED_REF)
        prepared_head = preparation.prepared_commit
        current_head = source_git.head()
        if current_head is None or source_git.status_lines():
            raise ValueError(
                f"The prepared publish repository changed at {source.root}. "
                "Review it before retrying."
            )
        if prepared_marker is None:
            if source_git.ref_sha(INTEGRATED_REF) == current_head:
                source_git.config_unset(PUBLISH_DECK_CONFIG)
                return
            source_git.update_ref(PUBLISH_PREPARED_REF, prepared_head)
        elif current_head != prepared_head:
            raise ValueError(
                f"The prepared publish repository changed at {source.root}. "
                "Review it before retrying."
            )
        plan = _prepare_publish_recovery_plan(collection_root, deck, source)
    else:
        plan = _prepare_publish_plan(collection_root, deck, source)
        _write_publish_files(plan)
        source_git.init_repo(initial_branch="main")
        source_git.copy_identity_from(collection_git)
        source_git.checkpoint(f"Publish collab deck {deck}")
        prepared_head = source_git.head()
        if prepared_head is None:
            raise ValueError(f"Could not prepare {source.display_name} for publish.")
        source_git.config_set(PUBLISH_DECK_CONFIG, deck)
        source_git.update_ref(PUBLISH_PREPARED_REF, prepared_head)

    github = GitHubHost(collection_root)
    source_git.set_remote("upstream", str(source.github_url))
    source_git.set_remote("publish", str(source.github_url))
    if github.repo_info(source.display_name) is None:
        github.create_repo(source.display_name)
    remote_head = _publish_remote_head(source_git, source, prepared_head)
    if remote_head is None:
        source_git.push("upstream", "HEAD", "main")

    source_git.checkout_or_create_branch(JOURNAL_BRANCH, prepared_head)
    source_git.run(["branch", "--unset-upstream"], check=False)
    source_git.update_ref(INTEGRATED_REF, prepared_head)
    _remove_published_local_files(plan)
    collection_git.commit_paths(
        [rendered.source_path for rendered in plan.files],
        f"Move {deck} into collab source {source.source_path}",
    )
    _transfer_sync_ownership(collection_root, plan)
    source_git.delete_ref(PUBLISH_PREPARED_REF)
    source_git.config_unset(PUBLISH_DECK_CONFIG)


def _publish_remote_head(
    repository: GitRepository,
    source: DeckSource,
    prepared_head: str,
) -> str | None:
    result = repository.run(["ls-remote", "upstream"], check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unreachable"
        raise ValueError(
            f"Could not verify GitHub repository {source.display_name}: {detail}"
        )
    refs = {
        ref: sha
        for line in result.stdout.splitlines()
        if "\t" in line
        for sha, ref in [line.split("\t", 1)]
    }
    if not refs:
        return None
    allowed_refs = {"HEAD", "refs/heads/main"}
    if (
        "refs/heads/main" in refs
        and set(refs).issubset(allowed_refs)
        and all(sha == prepared_head for sha in refs.values())
    ):
        return prepared_head
    raise RepositoryCollisionError(
        f"GitHub repository {source.display_name} contains unrelated content. "
        "Nothing was overwritten. Choose a new name."
    )


def _transfer_sync_ownership(collection_root: Path, plan: _PublishPlan) -> None:
    state = SyncState.open(collection_root)
    try:
        with state.write_tx():
            note_ids = state.resolve_note_ids(plan.note_keys)
            state.upsert_note_links(
                [(note_key, note_id) for note_key, note_id in note_ids.items()],
                source_path=plan.source.source_path,
            )
            for rendered in plan.files:
                deck_name = file_stem_to_deck_name(rendered.source_path.stem)
                deck_id = state.resolve_deck_id(deck_name)
                if deck_id is None:
                    continue
                state.upsert_deck(
                    deck_name,
                    deck_id,
                    source_path=plan.source.source_path,
                    md_path=str(rendered.target_path.relative_to(collection_root)),
                )
    finally:
        state.close()


def _select_decks(decks: tuple[ParsedDeck, ...], deck: str) -> list[ParsedDeck]:
    deck_filter = deck.strip()
    subdeck_scope = f"{deck_filter}::"
    selected = [
        parsed_deck
        for parsed_deck in decks
        if parsed_deck.deck_name == deck_filter
        or parsed_deck.deck_name.startswith(subdeck_scope)
    ]
    if not selected:
        raise ValueError(f"No local deck files found for '{deck}'")
    return selected


def _scoped_configs(
    source: DeckSource,
    configs: list[NoteType],
) -> list[NoteType]:
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def _prepare_publish_plan(
    collection_root: Path,
    deck: str,
    source: DeckSource,
) -> _PublishPlan:
    parsed_source = parse_source(DeckSource.local(collection_root))
    root_configs = list(parsed_source.note_types)
    root_config_by_name = {config.name: config for config in root_configs}

    selected_decks = _select_decks(parsed_source.decks, deck)
    note_types_used: set[str] = set()
    media_used: set[str] = set()

    for parsed_deck in selected_decks:
        for note in parsed_deck.parsed.notes:
            note_types_used.add(note.note_type)
            for field_value in note.fields.values():
                media_used.update(extract_media_references(field_value))

    note_keys = require_note_keys(deck.parsed for deck in selected_decks)

    media_paths = {
        media_name: collection_root / LOCAL_MEDIA_DIR / media_name
        for media_name in media_used
    }
    missing_media = sorted(
        media_name for media_name, path in media_paths.items() if not path.exists()
    )
    if missing_media:
        raise ValueError(
            "Cannot publish collab deck: referenced media file(s) missing: "
            + ", ".join(missing_media)
        )

    used_configs = [root_config_by_name[name] for name in sorted(note_types_used)]
    scoped_configs = _scoped_configs(source, used_configs)
    scoped_config_by_name = {config.name: config for config in scoped_configs}
    rendered_files: list[_RenderedPublishFile] = []
    for parsed_deck in selected_decks:
        for note in parsed_deck.parsed.notes:
            note.note_type = source.scope_note_type_name(note.note_type)
        rendered_files.append(
            _RenderedPublishFile(
                source_path=parsed_deck.path,
                target_path=source.root / parsed_deck.path.name,
                content=render_notes_to_markdown(
                    parsed_deck.parsed.notes,
                    scoped_config_by_name,
                ),
            )
        )

    return _PublishPlan(
        deck=deck,
        source=source,
        files=rendered_files,
        note_keys=note_keys,
        assets=_capture_publish_assets(
            collection_root,
            used_note_types=used_configs,
            media_used=media_used,
        ),
    )


def _prepare_publish_recovery_plan(
    collection_root: Path,
    deck: str,
    source: DeckSource,
) -> _PublishPlan:
    parsed_source = parse_source(source)
    selected_decks = _select_decks(parsed_source.decks, deck)
    rendered_files: list[_RenderedPublishFile] = []
    for parsed_deck in selected_decks:
        rendered_files.append(
            _RenderedPublishFile(
                source_path=collection_root / parsed_deck.path.name,
                target_path=parsed_deck.path,
                content=parsed_deck.parsed.raw_content,
            )
        )
    note_keys = require_note_keys(deck.parsed for deck in selected_decks)
    return _PublishPlan(
        deck=deck,
        source=source,
        files=rendered_files,
        note_keys=note_keys,
        assets=(),
    )


def _capture_publish_assets(
    collection_root: Path,
    *,
    used_note_types: list[NoteType],
    media_used: set[str],
) -> tuple[_PublishAsset, ...]:
    root = collection_root.absolute()
    note_types_dir = root / NOTE_TYPES_DIR
    paths = [
        note_types_dir / relative_path
        for note_type in used_note_types
        for relative_path in note_type.source_files
    ]
    media_dir = root / LOCAL_MEDIA_DIR
    paths.extend(media_dir / media_name for media_name in sorted(media_used))

    entries: dict[Path, _PublishAsset] = {}
    for path in paths:
        path = path.absolute()
        if not path.is_file():
            raise ValueError(
                f"Publish-applicable path '{path.relative_to(root)}' is not a "
                "regular file."
            )
        relative_path = path.relative_to(root)
        entry = _PublishAsset(
            relative_path=relative_path,
            content=path.read_bytes(),
        )
        entries[relative_path] = entry
    return tuple(
        entries[path] for path in sorted(entries, key=lambda item: item.as_posix())
    )


def _write_publish_files(plan: _PublishPlan) -> None:
    source = plan.source
    source.root.mkdir(parents=True, exist_ok=False)
    (source.root / LOCAL_MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    source.note_types_dir.mkdir(parents=True, exist_ok=True)

    readme_template = (
        resources.files("ankiops.collab")
        .joinpath("resources/README.md")
        .read_text(encoding="utf-8")
    )
    readme = source.root / "README.md"
    readme.write_text(
        readme_template.replace("{{DECK_NAME}}", plan.deck).replace(
            "{{REPOSITORY}}", source.display_name
        ),
        encoding="utf-8",
    )
    gitignore = source.root / ".gitignore"
    gitignore.write_text(
        resources.files("ankiops.collab")
        .joinpath("resources/.gitignore")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for rendered in plan.files:
        rendered.target_path.write_text(rendered.content, encoding="utf-8")

    for entry in plan.assets:
        target = source.root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.content)


def _remove_published_local_files(plan: _PublishPlan) -> None:
    for rendered in plan.files:
        rendered.source_path.unlink(missing_ok=True)

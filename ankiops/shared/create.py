"""Publish GitHub shared sources from local decks."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ankiops.collection import LOCAL_MEDIA_DIR, NOTE_TYPES_DIR, file_stem_to_deck_name
from ankiops.deck_sources import DeckSource
from ankiops.git import RepositoryGit
from ankiops.markdown import DeckFile, read_deck_file, render_notes_to_markdown
from ankiops.media import extract_media_references
from ankiops.note_types import NoteType, load_note_types
from ankiops.shared.errors import format_missing_note_keys_error
from ankiops.shared.hosting import GitHubHost
from ankiops.sync.state import SyncState


@dataclass(frozen=True)
class _RenderedCreateFile:
    source_path: Path
    target_path: Path
    content: str


@dataclass(frozen=True)
class _CreatePlan:
    source: DeckSource
    files: list[_RenderedCreateFile]
    note_types_used: set[str]
    media_used: set[str]
    note_keys: set[str]


def create_shared_deck(
    collection_dir: Path,
    deck: str,
    source: DeckSource,
    *,
    public: bool,
) -> None:
    root_repo = RepositoryGit(collection_dir)
    root_repo.ensure_repo("AnkiOps collections require a Git repository at the root.")
    existing_repo = RepositoryGit(source.root) if source.root.exists() else None
    if existing_repo is not None and not existing_repo.is_repo():
        raise ValueError(
            f"Shared source path exists but is not a Git repository: {source.root}"
        )
    if existing_repo is not None:
        try:
            _selected_deck_files(collection_dir, deck)
        except ValueError:
            return

    plan = _prepare_create_plan(collection_dir, deck, source)
    host = GitHubHost(collection_dir)
    host.create_repo(source.source_id, public=public)
    originals = {
        rendered.source_path: rendered.source_path.read_bytes()
        for rendered in plan.files
    }
    try:
        if existing_repo is None:
            _write_create_files(collection_dir, plan)
            shared_repo = RepositoryGit(source.root)
            shared_repo.init_repo(initial_branch="main")
            _copy_git_identity(root_repo, shared_repo)
            shared_repo.checkpoint(f"Publish shared deck {deck}")
        else:
            shared_repo = existing_repo
        shared_repo.set_remote("upstream", str(source.github_url))
        shared_repo.set_remote("publish", str(source.github_url))
        shared_repo.push("upstream", "HEAD", "main")
        shared_repo.ensure_work_branch("ankiops/work", "main")

        _unlink_created_source_files(plan)
        root_repo.commit_paths(
            [rendered.source_path for rendered in plan.files],
            f"Move {deck} into shared source {source.source_id}",
        )
        _transfer_sync_ownership(collection_dir, plan)
    except Exception:
        for path, content in originals.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        raise


def _copy_git_identity(root_repo: RepositoryGit, shared_repo: RepositoryGit) -> None:
    for key in ("user.name", "user.email"):
        value = root_repo.run(["config", "--get", key], check=False).stdout.strip()
        if value:
            shared_repo.run(["config", key, value])


def _transfer_sync_ownership(collection_dir: Path, plan: _CreatePlan) -> None:
    db = SyncState.open(collection_dir)
    try:
        note_ids = db.resolve_note_ids(plan.note_keys)
        db.upsert_note_links(
            [(note_key, note_id) for note_key, note_id in note_ids.items()],
            source_id=plan.source.source_id,
        )
        for rendered in plan.files:
            deck_name = file_stem_to_deck_name(rendered.source_path.stem)
            deck_id = db.resolve_deck_id(deck_name)
            if deck_id is None:
                continue
            db.upsert_deck(
                deck_name,
                deck_id,
                source_id=plan.source.source_id,
                md_path=str(rendered.target_path.relative_to(collection_dir)),
            )
    finally:
        db.close()


def _selected_deck_files(collection_dir: Path, deck: str) -> list[Path]:
    deck_filter = deck.strip()
    subdeck_scope = f"{deck_filter}::"
    files = []
    local_source = DeckSource.local(collection_dir)
    for md_file in local_source.deck_files():
        deck_name = file_stem_to_deck_name(md_file.stem)
        if deck_name == deck_filter or deck_name.startswith(subdeck_scope):
            files.append(md_file)
    if not files:
        raise ValueError(f"No local deck files found for '{deck}'")
    return files


def _scoped_configs(
    source: DeckSource,
    configs: list[NoteType],
) -> list[NoteType]:
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def _prepare_create_plan(
    collection_dir: Path,
    deck: str,
    source: DeckSource,
) -> _CreatePlan:
    root_configs = load_note_types(collection_dir / NOTE_TYPES_DIR)
    root_config_by_name = {config.name: config for config in root_configs}

    selected_files = _selected_deck_files(collection_dir, deck)
    parsed_files: list[tuple[Path, DeckFile]] = []
    note_types_used: set[str] = set()
    media_used: set[str] = set()
    note_keys: set[str] = set()
    missing_note_keys: list[str] = []

    for md_file in selected_files:
        parsed = read_deck_file(
            md_file,
            note_types=root_configs,
            context_root=collection_dir,
        )
        parsed_files.append((md_file, parsed))
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing_note_keys.append(f"{md_file.name} note {index}")
            elif note.note_key in note_keys:
                raise ValueError(
                    f"Duplicate note_key in shared publish set: {note.note_key}"
                )
            else:
                note_keys.add(note.note_key)
            note_types_used.add(note.note_type)
            for field_value in note.fields.values():
                media_used.update(extract_media_references(field_value))

    if missing_note_keys:
        raise ValueError(format_missing_note_keys_error(len(missing_note_keys)))

    unknown_note_types = sorted(note_types_used - set(root_config_by_name))
    if unknown_note_types:
        raise ValueError(
            "Unknown note type(s) while publishing shared deck: "
            + ", ".join(unknown_note_types)
        )

    missing_media = sorted(
        media_name
        for media_name in media_used
        if not (collection_dir / LOCAL_MEDIA_DIR / media_name).exists()
    )
    if missing_media:
        raise ValueError(
            "Cannot publish shared deck: referenced media file(s) missing: "
            + ", ".join(missing_media)
        )

    used_configs = [root_config_by_name[name] for name in sorted(note_types_used)]
    scoped_configs = _scoped_configs(source, used_configs)
    scoped_config_by_name = {config.name: config for config in scoped_configs}
    rendered_files: list[_RenderedCreateFile] = []
    for md_file, parsed in parsed_files:
        for note in parsed.notes:
            note.note_type = source.scope_note_type_name(note.note_type)
        rendered_files.append(
            _RenderedCreateFile(
                source_path=md_file,
                target_path=source.root / md_file.name,
                content=render_notes_to_markdown(parsed.notes, scoped_config_by_name),
            )
        )

    return _CreatePlan(
        source=source,
        files=rendered_files,
        note_types_used=note_types_used,
        media_used=media_used,
        note_keys=note_keys,
    )


def _styling_refs(note_type_dir: Path) -> list[str]:
    with open(note_type_dir / "note_type.yaml", encoding="utf-8") as file:
        info = yaml.safe_load(file) or {}
    styling = info.get("styling")
    if isinstance(styling, str):
        return [styling]
    if isinstance(styling, list):
        styling_refs: list[str] = []
        for css_file in styling:
            if not isinstance(css_file, str) or not css_file.strip():
                raise ValueError(
                    f"Note type '{note_type_dir.name}' has invalid styling entry "
                    f"'{css_file}'. Expected non-empty file names."
                )
            styling_refs.append(css_file.strip())
        if styling_refs:
            return styling_refs
    raise ValueError(
        f"Note type '{note_type_dir.name}' must reference at least one styling file."
    )


def _relative_note_type_asset_path(
    note_types_dir: Path,
    note_type: str,
    asset_path: Path,
    asset_ref: str,
) -> Path:
    try:
        return asset_path.resolve().relative_to(note_types_dir.resolve())
    except ValueError as error:
        raise ValueError(
            f"Note type '{note_type}' references styling file '{asset_ref}' "
            "outside the note_types directory."
        ) from error


def _copy_note_type_styling_assets(
    source_note_types_dir: Path,
    target_note_types_dir: Path,
    note_type: str,
) -> list[Path]:
    source_note_type_dir = source_note_types_dir / note_type
    touched: list[Path] = []

    for asset_ref in _styling_refs(source_note_type_dir):
        source_asset = source_note_type_dir / asset_ref
        if not source_asset.exists():
            raise ValueError(
                f"Note type '{note_type}' references missing styling file "
                f"'{asset_ref}'."
            )
        relative_asset = _relative_note_type_asset_path(
            source_note_types_dir,
            note_type,
            source_asset,
            asset_ref,
        )
        target_asset = target_note_types_dir / relative_asset
        target_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_asset, target_asset)
        touched.append(target_asset)

    return touched


def _write_create_files(collection_dir: Path, plan: _CreatePlan) -> list[Path]:
    source = plan.source
    source.root.mkdir(parents=True, exist_ok=False)
    (source.root / LOCAL_MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    source.note_types_dir.mkdir(parents=True, exist_ok=True)

    touched: list[Path] = [source.root]
    for rendered in plan.files:
        rendered.target_path.write_text(rendered.content, encoding="utf-8")
        touched.extend([rendered.target_path, rendered.source_path])

    for note_type in sorted(plan.note_types_used):
        src = collection_dir / NOTE_TYPES_DIR / note_type
        dst = source.note_types_dir / note_type
        shutil.copytree(src, dst, dirs_exist_ok=True)
        touched.append(dst)
        touched.extend(
            _copy_note_type_styling_assets(
                collection_dir / NOTE_TYPES_DIR,
                source.note_types_dir,
                note_type,
            )
        )

    root_media = collection_dir / LOCAL_MEDIA_DIR
    shared_media = source.root / LOCAL_MEDIA_DIR
    for media_name in sorted(plan.media_used):
        src = root_media / media_name
        dst = shared_media / media_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        touched.append(dst)

    return touched


def _unlink_created_source_files(plan: _CreatePlan) -> None:
    for rendered in plan.files:
        rendered.source_path.unlink()

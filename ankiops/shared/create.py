"""Create GitHub shared sources from local decks."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ankiops.config import LOCAL_MEDIA_DIR, NOTE_TYPES_DIR, file_stem_to_deck_name
from ankiops.export_notes import _render_notes_to_markdown
from ankiops.fs import FileSystemAdapter
from ankiops.git import CollectionGit
from ankiops.models import MarkdownFile, NoteTypeConfig
from ankiops.shared.hosting import ensure_create_repo
from ankiops.sources import SHARED_BRANCH, SyncSource
from ankiops.sync_media import _extract_media_references


@dataclass(frozen=True)
class _RenderedCreateFile:
    source_path: Path
    target_path: Path
    content: str


@dataclass(frozen=True)
class _CreatePlan:
    source: SyncSource
    files: list[_RenderedCreateFile]
    note_types_used: set[str]
    media_used: set[str]


def create_shared_deck(
    collection_dir: Path,
    deck: str,
    source: SyncSource,
    *,
    public: bool,
) -> None:
    repo = CollectionGit(collection_dir)
    repo.ensure_repo("Shared commands require a git-backed collection.")
    if source.root.exists():
        raise ValueError(f"Shared source already exists: {source.source_id}")

    plan = _prepare_create_plan(collection_dir, deck, source)
    repo.ensure_clean_index(
        "Shared create requires a clean git index. Commit or unstage "
        "existing changes before creating a shared source."
    )
    ensure_create_repo(repo, source, public=public)

    initial_head = repo.head()
    branch: str | None = None
    try:
        touched = _write_create_files(collection_dir, plan)
        repo.commit_create_move(
            touched_paths=touched,
            source_paths=[rendered.source_path for rendered in plan.files],
            message=f"AnkiOps: create {source.source_id} from {deck}",
        )
        branch = repo.subtree_split(source)
        if source.github_url:
            repo.push_ref(source.github_url, branch, SHARED_BRANCH)
    except Exception:
        _cleanup_failed_create(repo, plan, initial_head=initial_head, branch=branch)
        raise

    _unlink_created_source_files(plan)


def _cleanup_failed_create(
    repo: CollectionGit,
    plan: _CreatePlan,
    *,
    initial_head: str | None = None,
    branch: str | None = None,
) -> None:
    repo.rollback_to(initial_head)
    repo.delete_branch_if_exists(branch)
    create_paths = [
        repo.source_prefix(plan.source),
        *[repo.rel_path(rendered.source_path) for rendered in plan.files],
    ]
    repo.unstage_or_untrack(create_paths)
    _remove_create_source_root(plan)


def _remove_create_source_root(plan: _CreatePlan) -> None:
    if plan.source.root.exists():
        shutil.rmtree(plan.source.root)
    for parent in (plan.source.root.parent, plan.source.root.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            pass


def _selected_deck_files(collection_dir: Path, deck: str) -> list[Path]:
    deck_filter = deck.strip()
    subdeck_scope = f"{deck_filter}::"
    files = []
    for md_file in sorted(collection_dir.glob("*.md")):
        deck_name = file_stem_to_deck_name(md_file.stem)
        if deck_name == deck_filter or deck_name.startswith(subdeck_scope):
            files.append(md_file)
    if not files:
        raise ValueError(f"No local deck files found for '{deck}'")
    return files


def _scoped_configs(
    source: SyncSource,
    configs: list[NoteTypeConfig],
) -> list[NoteTypeConfig]:
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def _prepare_create_plan(
    collection_dir: Path,
    deck: str,
    source: SyncSource,
) -> _CreatePlan:
    fs = FileSystemAdapter()
    root_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    root_config_by_name = {config.name: config for config in root_configs}
    parser = FileSystemAdapter()
    parser.set_configs(root_configs)

    selected_files = _selected_deck_files(collection_dir, deck)
    parsed_files: list[tuple[Path, MarkdownFile]] = []
    note_types_used: set[str] = set()
    media_used: set[str] = set()
    note_keys: set[str] = set()
    missing_note_keys: list[str] = []

    for md_file in selected_files:
        parsed = parser.read_markdown_file(md_file, context_root=collection_dir)
        parsed_files.append((md_file, parsed))
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing_note_keys.append(f"{md_file.name} note {index}")
            elif note.note_key in note_keys:
                raise ValueError(
                    f"Duplicate note_key in shared create set: {note.note_key}"
                )
            else:
                note_keys.add(note.note_key)
            note_types_used.add(note.note_type)
            for field_value in note.fields.values():
                media_used.update(_extract_media_references(field_value))

    if missing_note_keys:
        raise ValueError(
            "Cannot create shared source: notes are missing note_key metadata: "
            + ", ".join(missing_note_keys)
            + ". Run 'ankiops ma' first or add explicit note_key comments."
        )

    unknown_note_types = sorted(note_types_used - set(root_config_by_name))
    if unknown_note_types:
        raise ValueError(
            "Unknown note type(s) while creating shared source: "
            + ", ".join(unknown_note_types)
        )

    missing_media = sorted(
        media_name
        for media_name in media_used
        if not (collection_dir / LOCAL_MEDIA_DIR / media_name).exists()
    )
    if missing_media:
        raise ValueError(
            "Cannot create shared source: referenced media file(s) missing: "
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
                content=_render_notes_to_markdown(parsed.notes, scoped_config_by_name),
            )
        )

    return _CreatePlan(
        source=source,
        files=rendered_files,
        note_types_used=note_types_used,
        media_used=media_used,
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

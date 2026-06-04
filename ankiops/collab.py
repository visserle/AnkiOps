"""GitHub-native collaboration source commands."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from ankiops.ankiops_bridge import AnkiOpsBridgeError
from ankiops.cli_anki import connect_or_exit
from ankiops.config import (
    LOCAL_MEDIA_DIR,
    NOTE_TYPES_DIR,
    file_stem_to_deck_name,
    require_collection_dir,
)
from ankiops.export_notes import _render_notes_to_markdown
from ankiops.fs import FileSystemAdapter
from ankiops.log import clickable_path
from ankiops.models import ANKIOPS_KEY_FIELD, MarkdownFile, NoteTypeConfig
from ankiops.sources import (
    COLLAB_BRANCH,
    SyncSource,
    discover_sync_sources,
    load_configs_for_source,
    markdown_files_for_source,
)
from ankiops.sync_media import _extract_media_references
from ankiops.sync_note_types import sync_note_type_configs

logger = logging.getLogger(__name__)

_SAFE_SLUG_PART_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$"
)
_GITHUB_OWNER_MAX_LENGTH = 39
_GITHUB_REPO_MAX_LENGTH = 100


@dataclass(frozen=True)
class _RenderedPublishFile:
    source_path: Path
    target_path: Path
    content: str


@dataclass(frozen=True)
class _PublishPlan:
    source: SyncSource
    deck_name: str
    deck_names: set[str]
    files: list[_RenderedPublishFile]
    scoped_configs: list[NoteTypeConfig]
    note_types_used: set[str]
    note_type_by_key: dict[str, str]
    media_used: set[str]
    note_keys: set[str]


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


def _source_prefix(collection_dir: Path, source: SyncSource) -> str:
    return str(source.root.relative_to(collection_dir))


def _run_git(collection_dir: Path, args: list[str]) -> subprocess.CompletedProcess:
    logger.debug("git %s", " ".join(args))
    return subprocess.run(
        ["git", *args],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=True,
    )


def _ensure_git_repo(collection_dir: Path) -> None:
    try:
        _run_git(collection_dir, ["rev-parse", "--git-dir"])
    except subprocess.CalledProcessError as error:
        raise ValueError("Collab commands require a git-backed collection.") from error


def _is_git_tracked(collection_dir: Path, rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel_path],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _git_commit_paths(collection_dir: Path, paths: list[Path], message: str) -> None:
    if not paths:
        return
    rel_paths = []
    seen = set()
    for path in paths:
        rel_path = str(path.relative_to(collection_dir))
        if rel_path in seen:
            continue
        if path.exists() or _is_git_tracked(collection_dir, rel_path):
            rel_paths.append(rel_path)
            seen.add(rel_path)
    if not rel_paths:
        return
    _run_git(collection_dir, ["add", "-A", "--", *rel_paths])
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *rel_paths],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if diff.returncode == 0:
        return
    _run_git(collection_dir, ["commit", "-m", message, "--", *rel_paths])


def _subtree_add(collection_dir: Path, source: SyncSource) -> None:
    if source.github_url is None:
        raise ValueError(f"Cannot derive GitHub URL for {source.display_name}")
    _run_git(
        collection_dir,
        [
            "subtree",
            "add",
            "--prefix",
            _source_prefix(collection_dir, source),
            source.github_url,
            COLLAB_BRANCH,
        ],
    )


def _subtree_pull(collection_dir: Path, source: SyncSource) -> None:
    if source.github_url is None:
        raise ValueError(f"Cannot derive GitHub URL for {source.display_name}")
    _run_git(
        collection_dir,
        [
            "subtree",
            "pull",
            "--prefix",
            _source_prefix(collection_dir, source),
            source.github_url,
            COLLAB_BRANCH,
        ],
    )


def _subtree_split(collection_dir: Path, source: SyncSource) -> str:
    branch = f"ankiops-{source.source_id.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
    _run_git(
        collection_dir,
        [
            "subtree",
            "split",
            "--prefix",
            _source_prefix(collection_dir, source),
            "-b",
            branch,
        ],
    )
    return branch


def _visibility_flag(public: bool) -> str:
    return "--public" if public else "--private"


def _manual_repo_create_command(source: SyncSource, *, public: bool) -> str:
    slug = source.github_slug or source.source_id
    return f"gh repo create {slug} {_visibility_flag(public)}"


def _github_repo_exists(collection_dir: Path, source: SyncSource) -> bool:
    slug = source.github_slug
    if slug is None or source.github_url is None:
        raise ValueError(f"Cannot derive GitHub repo for {source.display_name}")

    gh_path = shutil.which("gh")
    if gh_path:
        result = subprocess.run(
            [gh_path, "repo", "view", slug],
            cwd=collection_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    result = subprocess.run(
        ["git", "ls-remote", source.github_url],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _create_github_repo(
    collection_dir: Path,
    source: SyncSource,
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
        cwd=collection_dir,
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


def _ensure_publish_repo(
    collection_dir: Path,
    source: SyncSource,
    *,
    public: bool,
) -> None:
    if _github_repo_exists(collection_dir, source):
        return
    if shutil.which("gh") is None:
        raise ValueError(
            "GitHub repository does not exist or is not accessible: "
            f"{source.github_slug}. Create it first with: "
            f"{_manual_repo_create_command(source, public=public)}"
        )
    _create_github_repo(collection_dir, source, public=public)


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


def _deck_names_for_files(files: list[Path]) -> set[str]:
    return {file_stem_to_deck_name(md_file.stem) for md_file in files}


def _scoped_configs(
    source: SyncSource,
    configs: list[NoteTypeConfig],
) -> list[NoteTypeConfig]:
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def _prepare_publish_plan(
    collection_dir: Path,
    deck: str,
    source: SyncSource,
) -> _PublishPlan:
    fs = FileSystemAdapter()
    root_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    root_config_by_name = {config.name: config for config in root_configs}
    parser = FileSystemAdapter()
    parser.set_configs(root_configs)

    selected_files = _selected_deck_files(collection_dir, deck)
    deck_names = _deck_names_for_files(selected_files)
    parsed_files: list[tuple[Path, MarkdownFile]] = []
    note_types_used: set[str] = set()
    note_type_by_key: dict[str, str] = {}
    media_used: set[str] = set()
    note_keys: set[str] = set()

    for md_file in selected_files:
        parsed = parser.read_markdown_file(md_file, context_root=collection_dir)
        parsed_files.append((md_file, parsed))
        for note in parsed.notes:
            if note.note_key:
                if note.note_key in note_keys:
                    raise ValueError(
                        f"Duplicate note_key in publish set: {note.note_key}"
                    )
                note_keys.add(note.note_key)
                note_type_by_key[note.note_key] = note.note_type
            note_types_used.add(note.note_type)
            for field_value in note.fields.values():
                media_used.update(_extract_media_references(field_value))

    unknown_note_types = sorted(note_types_used - set(root_config_by_name))
    if unknown_note_types:
        raise ValueError(
            "Unknown note type(s) while publishing: "
            + ", ".join(unknown_note_types)
        )

    missing_media = sorted(
        media_name
        for media_name in media_used
        if not (collection_dir / LOCAL_MEDIA_DIR / media_name).exists()
    )
    if missing_media:
        raise ValueError(
            "Cannot publish: referenced media file(s) missing: "
            + ", ".join(missing_media)
        )

    used_configs = [
        root_config_by_name[name]
        for name in sorted(note_types_used)
    ]
    scoped_configs = _scoped_configs(source, used_configs)
    scoped_config_by_name = {config.name: config for config in scoped_configs}
    rendered_files: list[_RenderedPublishFile] = []
    for md_file, parsed in parsed_files:
        for note in parsed.notes:
            note.note_type = source.scope_note_type_name(note.note_type)
        rendered_files.append(
            _RenderedPublishFile(
                source_path=md_file,
                target_path=source.root / md_file.name,
                content=_render_notes_to_markdown(parsed.notes, scoped_config_by_name),
            )
        )

    return _PublishPlan(
        source=source,
        deck_name=deck,
        deck_names=deck_names,
        files=rendered_files,
        scoped_configs=scoped_configs,
        note_types_used=note_types_used,
        note_type_by_key=note_type_by_key,
        media_used=media_used,
        note_keys=note_keys,
    )


def _collect_notes_to_convert(anki, plan: _PublishPlan) -> dict[str, list[int]]:
    note_ids = anki.fetch_all_note_ids(sorted(anki.fetch_model_names()))
    notes = anki.fetch_notes_info(note_ids)
    card_ids = [
        card_id
        for note in notes.values()
        for card_id in note.card_ids
    ]
    cards = anki.fetch_cards_info(card_ids)
    notes_to_convert: dict[str, list[int]] = {}

    for note in notes.values():
        deck_names = {
            card["deckName"]
            for card_id in note.card_ids
            if (card := cards.get(card_id)) and card.get("deckName")
        }
        if not deck_names.intersection(plan.deck_names):
            continue
        outside_decks = deck_names - plan.deck_names
        if outside_decks:
            raise ValueError(
                f"Cannot publish note {note.note_id}: it has cards both inside "
                f"the published deck tree and outside it ({', '.join(outside_decks)})."
            )
        note_key = note.fields.get(ANKIOPS_KEY_FIELD.name, "").strip()
        if not note_key:
            raise ValueError(
                f"Cannot publish note {note.note_id}: it has no AnkiOps Key. "
                "Run 'ankiops am' first so the note can be tracked."
            )
        if note_key not in plan.note_keys:
            raise ValueError(
                f"Cannot publish note {note.note_id}: its AnkiOps Key "
                f"'{note_key}' is not present in the local Markdown files. "
                "Run 'ankiops am' before publishing."
            )
        markdown_note_type = plan.note_type_by_key[note_key]
        target_note_type = plan.source.scope_note_type_name(markdown_note_type)
        if note.note_type == target_note_type:
            continue
        if note.note_type != markdown_note_type:
            raise ValueError(
                f"Cannot publish note {note.note_id}: Markdown would publish "
                f"note_key '{note_key}' as '{target_note_type}', but Anki "
                f"already has '{note.note_type}'. Use the same owner/repo slug "
                "as the existing collab source, or intentionally convert the "
                "Anki note type first."
            )
        notes_to_convert.setdefault(markdown_note_type, []).append(note.note_id)

    return notes_to_convert


def _convert_publish_notes(anki, plan: _PublishPlan) -> None:
    notes_to_convert = _collect_notes_to_convert(anki, plan)
    if not notes_to_convert:
        return

    sync_note_type_configs(anki, plan.scoped_configs, db_port=None)
    try:
        for old_model, note_ids in sorted(notes_to_convert.items()):
            anki.change_notes_notetype(
                sorted(note_ids),
                old_model,
                plan.source.scope_note_type_name(old_model),
            )
    except AnkiOpsBridgeError as error:
        raise ValueError(str(error)) from error


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


def _write_publish_files(collection_dir: Path, plan: _PublishPlan) -> list[Path]:
    source = plan.source
    source.root.mkdir(parents=True, exist_ok=False)
    (source.root / LOCAL_MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    source.note_types_dir.mkdir(parents=True, exist_ok=True)

    touched: list[Path] = [source.root]
    for rendered in plan.files:
        rendered.target_path.write_text(rendered.content, encoding="utf-8")
        rendered.source_path.unlink()
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
    collab_media = source.root / LOCAL_MEDIA_DIR
    for media_name in sorted(plan.media_used):
        src = root_media / media_name
        dst = collab_media / media_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        touched.append(dst)

    return touched


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


def _open_pr_if_possible(collection_dir: Path, source: SyncSource, branch: str) -> None:
    if source.github_url is None:
        logger.info("Prepared branch %s. Push it and open a PR manually.", branch)
        return
    if shutil.which("gh") is None:
        logger.info(
            "Prepared branch %s. Push it to %s and open a PR manually.",
            branch,
            source.github_url,
        )
        return
    push = subprocess.run(
        ["git", "push", source.github_url, f"{branch}:{branch}"],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if push.returncode != 0:
        logger.info(
            "Prepared branch %s. Push it to %s and open a PR manually.",
            branch,
            source.github_url,
        )
        logger.debug(push.stderr)
        return
    slug = source.github_slug
    gh = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            slug or "",
            "--head",
            branch,
            "--base",
            COLLAB_BRANCH,
            "--fill",
        ],
        cwd=collection_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if gh.returncode == 0:
        logger.info(gh.stdout.strip())
    else:
        logger.info("Pushed branch %s. Open the PR manually.", branch)
        logger.debug(gh.stderr)


def run_publish(args) -> None:
    collection_dir = require_collection_dir()
    source = _source_for_slug(collection_dir, args.repo)
    _ensure_git_repo(collection_dir)
    if source.root.exists():
        raise ValueError(f"Collab source already exists: {source.source_id}")

    plan = _prepare_publish_plan(collection_dir, args.deck, source)
    _ensure_publish_repo(
        collection_dir,
        source,
        public=bool(getattr(args, "public", False)),
    )
    anki = connect_or_exit()
    _convert_publish_notes(anki, plan)
    touched = _write_publish_files(collection_dir, plan)
    _git_commit_paths(
        collection_dir,
        touched,
        f"AnkiOps: publish {args.deck} to {source.source_id}",
    )
    branch = _subtree_split(collection_dir, source)
    if source.github_url:
        _run_git(
            collection_dir,
            ["push", source.github_url, f"{branch}:{COLLAB_BRANCH}"],
        )
    logger.info("Published %s to %s", args.deck, source.source_id)


def run_subscribe(args) -> None:
    collection_dir = require_collection_dir()
    source = _source_for_slug(collection_dir, args.repo)
    _ensure_git_repo(collection_dir)
    if source.root.exists():
        raise ValueError(f"Collab source already exists: {source.source_id}")
    _subtree_add(collection_dir, source)
    logger.info("Subscribed to %s at %s", args.repo, clickable_path(source.root))


def run_pull(args) -> None:
    collection_dir = require_collection_dir()
    if args.repo:
        source = _source_for_slug(collection_dir, args.repo)
    _ensure_git_repo(collection_dir)
    sources = discover_sync_sources(collection_dir)
    if args.repo:
        if not source.root.exists():
            raise ValueError(f"Unknown collab source: {source.source_id}")
        targets = [source]
    else:
        targets = [source for source in sources if source.is_collab]
    if not targets:
        logger.info("No collab sources found.")
        return
    for source in targets:
        _subtree_pull(collection_dir, source)
        logger.info("Pulled %s", source.source_id)


def run_contribute(args) -> None:
    collection_dir = require_collection_dir()
    source = _source_for_slug(collection_dir, args.repo)
    _ensure_git_repo(collection_dir)
    if not source.root.exists():
        raise ValueError(f"Unknown collab source: {source.source_id}")
    _ensure_contributable_note_keys(source)
    _git_commit_paths(
        collection_dir,
        [source.root],
        f"AnkiOps: contribute {source.source_id}",
    )
    branch = _subtree_split(collection_dir, source)
    _open_pr_if_possible(collection_dir, source, branch)


def run_status(args) -> None:
    collection_dir = require_collection_dir()
    sources = [
        source
        for source in discover_sync_sources(collection_dir)
        if source.is_collab
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

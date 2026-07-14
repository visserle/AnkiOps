"""Publish local decks as GitHub collab sources."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

import yaml

from ankiops.collab.errors import (
    RepositoryCollisionError,
    RepositoryCreationUncertainError,
    format_missing_note_keys_error,
)
from ankiops.collab.hosting import GitHubHost
from ankiops.collection import LOCAL_MEDIA_DIR, NOTE_TYPES_DIR, file_stem_to_deck_name
from ankiops.deck_sources import DeckSource
from ankiops.git import GitRepository
from ankiops.markdown import DeckFile, read_deck_file, render_notes_to_markdown
from ankiops.media import extract_media_references
from ankiops.note_types import NoteType, load_note_types
from ankiops.sync.state import SyncState


@dataclass(frozen=True)
class _RenderedPublishFile:
    source_path: Path
    target_path: Path
    content: str


@dataclass(frozen=True)
class _PublishManifestEntry:
    role: str
    relative_path: Path
    file_type: str
    content: bytes


@dataclass(frozen=True)
class _PublishManifest:
    entries: tuple[_PublishManifestEntry, ...]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for entry in self.entries:
            for value in (
                entry.role.encode(),
                entry.relative_path.as_posix().encode(),
                entry.file_type.encode(),
                entry.content,
            ):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return digest.hexdigest()


@dataclass(frozen=True)
class _PublishPlan:
    deck: str
    source: DeckSource
    files: list[_RenderedPublishFile]
    note_keys: set[str]
    manifest: _PublishManifest | None


def publish_collab_deck(
    collection_root: Path,
    deck: str,
    source: DeckSource,
    *,
    retry: bool = False,
    expected_prepared_head: str | None = None,
    expected_root_fingerprint: str | None = None,
    record_prepared: Callable[[str, str], None] | None = None,
) -> None:
    collection_git = GitRepository(collection_root)
    collection_git.ensure_repo(
        "AnkiOps collections require a Git repository at the root."
    )
    source_git = GitRepository(source.root) if source.root.exists() else None
    if source_git is not None and not source_git.is_repo():
        raise ValueError(
            f"Collab source path exists but is not a Git repository: {source.root}"
        )
    if source_git is not None and not retry:
        raise RepositoryCollisionError(
            f"Local collab source already exists: {source.display_name}. "
            "Choose a new name; the existing files were left untouched."
        )
    if source_git is not None and expected_prepared_head is None:
        raise RepositoryCollisionError(
            "The local collab source was not prepared by this publish: "
            f"{source.display_name}. Choose a new name; the existing files were "
            "left untouched."
        )
    if source_git is not None and (
        source_git.head() != expected_prepared_head or source_git.status_lines()
    ):
        raise ValueError(
            f"The interrupted publish source changed at {source.root}. "
            "Review those files before retrying."
        )
    recovery_plan = False
    try:
        _selected_deck_files(collection_root, deck)
    except ValueError:
        if source_git is None or not retry:
            raise
        if source_git.status_lines():
            raise ValueError(
                f"The interrupted publish source changed at {source.root}. "
                "Review those files before retrying."
            )
        plan = _prepare_publish_recovery_plan(collection_root, deck, source)
        recovery_plan = True
    else:
        plan = _prepare_publish_plan(collection_root, deck, source)
    github = GitHubHost(collection_root)
    root_fingerprint = (
        plan.manifest.fingerprint()
        if plan.manifest is not None
        else expected_root_fingerprint or ""
    )
    if (
        source_git is not None
        and not recovery_plan
        and root_fingerprint != expected_root_fingerprint
    ):
        raise ValueError(
            "The selected local deck files changed since this interrupted publish "
            "was prepared, or other publish-applicable files changed since "
            "preparation. The newer files were left untouched."
        )
    prepared_before_retry = source_git is not None
    prepared_this_attempt = source_git is None
    prepared_head = expected_prepared_head
    repository_created = False
    repository_creation_uncertain = False
    root_removal_started = False
    try:
        if source_git is None:
            _write_publish_files(plan)
            source_git = GitRepository(source.root)
            source_git.init_repo(initial_branch="main")
            _copy_git_identity(collection_git, source_git)
            source_git.checkpoint(f"Publish collab deck {deck}")
            prepared_head = source_git.head()
            if prepared_head is None:
                raise ValueError(
                    f"Could not prepare the collab source {source.display_name}."
                )
            if record_prepared is not None:
                record_prepared(prepared_head, root_fingerprint)
        source_git.set_remote("upstream", str(source.github_url))
        source_git.set_remote("publish", str(source.github_url))
        retrying_existing_remote = bool(
            retry
            and prepared_before_retry
            and _remote_is_reachable(source_git, "upstream")
        )
        if retrying_existing_remote:
            _ensure_retry_remote_matches(source_git, source)
        else:
            try:
                github.create_repo(source.display_name)
            except RepositoryCreationUncertainError:
                repository_creation_uncertain = True
                raise
            repository_created = True
        source_git.push("upstream", "HEAD", "main")
        if retrying_existing_remote:
            _ensure_retry_remote_matches(source_git, source)
        source_git.checkout_or_create_branch("ankiops/journal", "main")
        source_git.run(["branch", "--unset-upstream"], check=False)
        source_git.update_ref("refs/ankiops/integrated", "main")

        if plan.manifest is not None and not _publish_manifest_is_current(
            collection_root, deck, source, plan.manifest
        ):
            raise ValueError(
                "The selected local deck files changed, or other publish-applicable "
                "files changed, while the publish was prepared. The newer files "
                "were left untouched."
            )
        root_removal_started = True
        _remove_published_local_files(plan)
        collection_git.commit_paths(
            [rendered.source_path for rendered in plan.files],
            f"Move {deck} into collab source {source.source_path}",
        )
        if not _published_root_is_ready_for_ownership(plan):
            raise ValueError(
                "The selected local deck files changed, or other publish-applicable "
                "files changed, during the final handoff. The newer files were "
                "left untouched."
            )
        _transfer_sync_ownership(collection_root, plan)
    except RepositoryCollisionError as error:
        if (
            source_git is not None
            and prepared_head is not None
            and _prepared_source_is_unchanged(source_git, prepared_head)
            and plan.manifest is not None
            and _publish_manifest_is_current(
                collection_root, deck, source, plan.manifest
            )
        ):
            shutil.rmtree(source.root)
            raise
        raise RepositoryCollisionError(
            f"{error} The prepared files were preserved at {source.root}; deleting "
            "them could remove the only safe copy."
        ) from error
    except Exception:
        for entry in plan.manifest.entries if plan.manifest is not None else ():
            path = collection_root / entry.relative_path
            if root_removal_started and entry.role == "deck" and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(entry.content)
        if (
            prepared_this_attempt
            and not repository_created
            and not repository_creation_uncertain
        ):
            shutil.rmtree(source.root, ignore_errors=True)
        raise


def _remote_is_reachable(repository: GitRepository, remote: str) -> bool:
    return repository.run(["ls-remote", remote], check=False).returncode == 0


def _prepared_source_is_unchanged(
    repository: GitRepository,
    prepared_head: str,
) -> bool:
    if repository.head() != prepared_head:
        return False
    return not repository.run(
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored",
        ]
    ).stdout


def _publish_manifest_is_current(
    collection_root: Path,
    deck: str,
    source: DeckSource,
    expected: _PublishManifest,
) -> bool:
    try:
        current = _prepare_publish_plan(collection_root, deck, source).manifest
    except (OSError, ValueError):
        return False
    return current == expected


def _manifest_entry_is_current(
    collection_root: Path, entry: _PublishManifestEntry
) -> bool:
    path = collection_root / entry.relative_path
    if entry.file_type != "regular" or path.is_symlink() or not path.is_file():
        return False
    try:
        return path.read_bytes() == entry.content
    except OSError:
        return False


def _published_root_is_ready_for_ownership(plan: _PublishPlan) -> bool:
    if plan.manifest is None:
        return True
    root = plan.source.collection_root
    for entry in plan.manifest.entries:
        path = root / entry.relative_path
        if entry.role == "deck":
            if path.exists() or path.is_symlink():
                return False
        elif not _manifest_entry_is_current(root, entry):
            return False
    try:
        return not _selected_deck_files(root, plan.deck)
    except ValueError as error:
        return str(error).startswith("No local deck files found for")


def _ensure_retry_remote_matches(
    repository: GitRepository,
    source: DeckSource,
) -> None:
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
        return
    local_head = repository.head()
    allowed_refs = {"HEAD", "refs/heads/main"}
    if (
        local_head is not None
        and "refs/heads/main" in refs
        and set(refs).issubset(allowed_refs)
        and all(sha == local_head for sha in refs.values())
    ):
        return
    raise RepositoryCollisionError(
        f"GitHub repository {source.display_name} contains unrelated content. "
        "Nothing was overwritten. Choose a new name."
    )


def _copy_git_identity(
    collection_git: GitRepository,
    source_git: GitRepository,
) -> None:
    for key in ("user.name", "user.email"):
        value = collection_git.run(["config", "--get", key], check=False).stdout.strip()
        if value:
            source_git.run(["config", key, value])


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


def _selected_deck_files(collection_root: Path, deck: str) -> list[Path]:
    return _select_deck_files(DeckSource.local(collection_root).deck_files(), deck)


def _select_deck_files(deck_files: list[Path], deck: str) -> list[Path]:
    deck_filter = deck.strip()
    subdeck_scope = f"{deck_filter}::"
    files = []
    for md_file in deck_files:
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


def _prepare_publish_plan(
    collection_root: Path,
    deck: str,
    source: DeckSource,
) -> _PublishPlan:
    root_configs = load_note_types(collection_root / NOTE_TYPES_DIR)
    root_config_by_name = {config.name: config for config in root_configs}

    selected_files = _selected_deck_files(collection_root, deck)
    parsed_files: list[tuple[Path, DeckFile]] = []
    note_types_used: set[str] = set()
    media_used: set[str] = set()
    note_keys: set[str] = set()
    missing_note_keys: list[str] = []

    for md_file in selected_files:
        md_file = _contained_regular_file(
            collection_root,
            md_file,
            label=f"Selected deck file '{md_file.name}'",
            scope="collection root",
        )
        parsed = read_deck_file(
            md_file,
            note_types=root_configs,
            context_root=collection_root,
        )
        parsed_files.append((md_file, parsed))
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing_note_keys.append(f"{md_file.name} note {index}")
            elif note.note_key in note_keys:
                raise ValueError(
                    f"Duplicate note_key in collab publish set: {note.note_key}"
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
            "Unknown note type(s) while publishing collab deck: "
            + ", ".join(unknown_note_types)
        )

    media_paths = {
        media_name: _contained_media_path(collection_root / LOCAL_MEDIA_DIR, media_name)
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
    for md_file, parsed in parsed_files:
        for note in parsed.notes:
            note.note_type = source.scope_note_type_name(note.note_type)
        rendered_files.append(
            _RenderedPublishFile(
                source_path=md_file,
                target_path=source.root / md_file.name,
                content=render_notes_to_markdown(parsed.notes, scoped_config_by_name),
            )
        )

    return _PublishPlan(
        deck=deck,
        source=source,
        files=rendered_files,
        note_keys=note_keys,
        manifest=_capture_publish_manifest(
            collection_root,
            selected_files=selected_files,
            note_types_used=note_types_used,
            media_used=media_used,
        ),
    )


def _prepare_publish_recovery_plan(
    collection_root: Path,
    deck: str,
    source: DeckSource,
) -> _PublishPlan:
    selected_files = _select_deck_files(source.deck_files(), deck)
    configs = _scoped_configs(source, load_note_types(source.note_types_dir))
    note_keys: set[str] = set()
    rendered_files: list[_RenderedPublishFile] = []
    missing_note_keys = 0
    for md_file in selected_files:
        parsed = read_deck_file(
            md_file,
            note_types=configs,
            context_root=source.root,
        )
        for note in parsed.notes:
            if not note.note_key:
                missing_note_keys += 1
            elif note.note_key in note_keys:
                raise ValueError(
                    f"Duplicate note_key in interrupted collab publish: {note.note_key}"
                )
            else:
                note_keys.add(note.note_key)
        rendered_files.append(
            _RenderedPublishFile(
                source_path=collection_root / md_file.name,
                target_path=md_file,
                content=md_file.read_text(encoding="utf-8"),
            )
        )
    if missing_note_keys:
        raise ValueError(format_missing_note_keys_error(missing_note_keys))
    return _PublishPlan(
        deck=deck,
        source=source,
        files=rendered_files,
        note_keys=note_keys,
        manifest=None,
    )


def _contained_media_path(media_dir: Path, media_name: str) -> Path:
    normalized_name = media_name.replace("\\", "/")
    media_prefix = f"{LOCAL_MEDIA_DIR}/"
    if normalized_name.startswith(media_prefix):
        normalized_name = normalized_name[len(media_prefix) :]
    return _contained_regular_file(
        media_dir,
        media_dir / normalized_name,
        label=f"Referenced media path '{media_name}'",
        scope="media directory",
    )


def _contained_regular_file(
    root: Path,
    candidate: Path,
    *,
    label: str,
    scope: str,
) -> Path:
    """Return a contained regular path without following any symlink component."""
    lexical_root = Path(os.path.abspath(root))
    lexical_candidate = Path(os.path.abspath(candidate))
    resolved_root = lexical_root.resolve()
    resolved_candidate = lexical_candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} points outside the {scope}.") from error
    try:
        relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{label} points outside the {scope}.") from error

    current = lexical_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"{label} is or traverses a symbolic link. Publish regular files only."
            )
    return lexical_candidate


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


def _template_refs(note_type_dir: Path) -> list[str]:
    with open(note_type_dir / "note_type.yaml", encoding="utf-8") as file:
        info = yaml.safe_load(file) or {}
    templates = info.get("templates")
    if templates is None:
        refs = ["Front.template.anki", "Back.template.anki"]
        template_index = 2
        while True:
            front = f"Front{template_index}.template.anki"
            back = f"Back{template_index}.template.anki"
            if (
                not (note_type_dir / front).exists()
                or not (note_type_dir / back).exists()
            ):
                break
            refs.extend([front, back])
            template_index += 1
        return refs
    return [
        str(template[side]).strip()
        for template in templates
        for side in ("front", "back")
    ]


def _relative_note_type_asset_path(
    note_types_dir: Path,
    note_type: str,
    asset_path: Path,
    asset_ref: str,
) -> Path:
    contained = _contained_regular_file(
        note_types_dir,
        asset_path,
        label=f"Note type '{note_type}' asset '{asset_ref}'",
        scope="note_types directory",
    )
    return contained.relative_to(Path(os.path.abspath(note_types_dir)))


def _note_type_manifest_paths(note_types_dir: Path, note_type: str) -> list[Path]:
    source_note_type_dir = note_types_dir / note_type

    asset_refs = [
        "note_type.yaml",
        *_template_refs(source_note_type_dir),
        *_styling_refs(source_note_type_dir),
    ]
    copied: set[Path] = set()
    paths: list[Path] = []
    for asset_ref in asset_refs:
        source_asset = source_note_type_dir / asset_ref
        if not source_asset.exists() or not source_asset.is_file():
            raise ValueError(
                f"Note type '{note_type}' references missing asset '{asset_ref}'."
            )
        relative_asset = _relative_note_type_asset_path(
            note_types_dir,
            note_type,
            source_asset,
            asset_ref,
        )
        if relative_asset in copied:
            continue
        copied.add(relative_asset)
        paths.append(note_types_dir / relative_asset)

    return paths


def _capture_publish_manifest(
    collection_root: Path,
    *,
    selected_files: list[Path],
    note_types_used: set[str],
    media_used: set[str],
) -> _PublishManifest:
    root = Path(os.path.abspath(collection_root))
    paths: list[tuple[str, Path]] = [("deck", path) for path in selected_files]
    note_types_dir = root / NOTE_TYPES_DIR
    for note_type in sorted(note_types_used):
        paths.extend(
            ("note_type", path)
            for path in _note_type_manifest_paths(note_types_dir, note_type)
        )
    media_dir = root / LOCAL_MEDIA_DIR
    paths.extend(
        ("media", _contained_media_path(media_dir, media_name))
        for media_name in sorted(media_used)
    )

    entries: dict[Path, _PublishManifestEntry] = {}
    for role, path in paths:
        path = Path(os.path.abspath(path))
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Publish-applicable path '{path.relative_to(root)}' is not a "
                "regular file."
            )
        relative_path = path.relative_to(root)
        entry = _PublishManifestEntry(
            role=role,
            relative_path=relative_path,
            file_type="regular",
            content=path.read_bytes(),
        )
        existing = entries.get(relative_path)
        if existing is not None and existing != entry:
            raise ValueError(
                f"Publish-applicable path '{relative_path}' has conflicting roles."
            )
        entries[relative_path] = entry
    return _PublishManifest(
        tuple(
            entries[path] for path in sorted(entries, key=lambda item: item.as_posix())
        )
    )


def _write_publish_files(plan: _PublishPlan) -> list[Path]:
    source = plan.source
    source.root.mkdir(parents=True, exist_ok=False)
    (source.root / LOCAL_MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    source.note_types_dir.mkdir(parents=True, exist_ok=True)

    touched: list[Path] = [source.root]
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
    touched.append(readme)
    gitignore = source.root / ".gitignore"
    gitignore.write_text(
        resources.files("ankiops.collab")
        .joinpath("resources/.gitignore")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    touched.append(gitignore)
    for rendered in plan.files:
        rendered.target_path.write_text(rendered.content, encoding="utf-8")
        touched.extend([rendered.target_path, rendered.source_path])

    if plan.manifest is None:
        raise ValueError("Cannot write a collab source without its publish manifest.")
    for entry in plan.manifest.entries:
        if entry.role == "deck":
            continue
        target = source.root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.content)
        touched.append(target)

    return touched


def _remove_published_local_files(plan: _PublishPlan) -> None:
    if plan.manifest is None:
        return
    entries = {entry.relative_path: entry for entry in plan.manifest.entries}
    for rendered in plan.files:
        relative_path = rendered.source_path.relative_to(plan.source.collection_root)
        entry = entries.get(relative_path)
        if entry is None or not _manifest_entry_is_current(
            plan.source.collection_root, entry
        ):
            raise ValueError(
                "The publish-applicable files changed during the final handoff. "
                "The newer files were left untouched."
            )
        rendered.source_path.unlink(missing_ok=True)

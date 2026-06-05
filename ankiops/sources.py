"""Sync source discovery for local and GitHub-collab decks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.models import NoteTypeConfig

COLLAB_DIR = "collab"
COLLAB_BRANCH = "main"
RESERVED_COLLAB_MARKDOWN = {
    "README.md",
    "README.markdown",
    "LICENSE.md",
    "CHANGELOG.md",
}


@dataclass(frozen=True)
class SyncSource:
    """A filesystem root that contributes decks to one logical collection."""

    root: Path
    source_id: str
    note_types_dir: Path
    is_collab: bool = False

    @classmethod
    def local(cls, collection_dir: Path, note_types_dir: Path | None = None):
        return cls(
            root=collection_dir,
            source_id="",
            note_types_dir=note_types_dir or collection_dir / NOTE_TYPES_DIR,
            is_collab=False,
        )

    @classmethod
    def collab(cls, collection_dir: Path, owner: str, repo: str):
        root = collection_dir / COLLAB_DIR / owner / repo
        return cls(
            root=root,
            source_id=f"{COLLAB_DIR}/{owner}/{repo}",
            note_types_dir=root / NOTE_TYPES_DIR,
            is_collab=True,
        )

    @property
    def display_name(self) -> str:
        return self.source_id or "local"

    @property
    def github_slug(self) -> str | None:
        if not self.is_collab:
            return None
        parts = self.source_id.split("/")
        if len(parts) != 3:
            return None
        return f"{parts[1]}/{parts[2]}"

    @property
    def github_url(self) -> str | None:
        slug = self.github_slug
        return f"https://github.com/{slug}.git" if slug else None

    def scope_note_type_name(self, name: str) -> str:
        if not self.is_collab:
            return name
        prefix = f"{self.source_id}/"
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def unscoped_note_type_name(self, name: str) -> str:
        if not self.is_collab:
            return name
        prefix = f"{self.source_id}/"
        return name[len(prefix) :] if name.startswith(prefix) else name


@dataclass(frozen=True)
class SourceConfigs:
    source: SyncSource
    configs: list[NoteTypeConfig]


def is_reserved_collab_markdown(path: Path) -> bool:
    name = path.name
    return name in RESERVED_COLLAB_MARKDOWN or name.startswith("_")


def markdown_files_for_source(source: SyncSource) -> list[Path]:
    files = sorted(source.root.glob("*.md"))
    if not source.is_collab:
        return files
    return [path for path in files if not is_reserved_collab_markdown(path)]


def discover_sync_sources(
    collection_dir: Path,
    *,
    note_types_dir: Path | None = None,
) -> list[SyncSource]:
    sources = [SyncSource.local(collection_dir, note_types_dir)]
    collab_root = collection_dir / COLLAB_DIR
    if not collab_root.exists():
        return sources

    for owner_dir in sorted(collab_root.iterdir(), key=lambda path: path.name):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir(), key=lambda path: path.name):
            if not repo_dir.is_dir():
                continue
            sources.append(
                SyncSource.collab(collection_dir, owner_dir.name, repo_dir.name)
            )
    return sources


def load_configs_for_source(source: SyncSource) -> list[NoteTypeConfig]:
    configs = FileSystemAdapter().load_note_type_configs(source.note_types_dir)
    if not source.is_collab:
        return configs
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def load_configs_for_sources(sources: list[SyncSource]) -> list[SourceConfigs]:
    return [
        SourceConfigs(source=source, configs=load_configs_for_source(source))
        for source in sources
    ]

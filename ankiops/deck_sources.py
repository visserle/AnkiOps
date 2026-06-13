"""Deck source discovery for local and GitHub-shared decks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ankiops.collection import NOTE_TYPES_DIR
from ankiops.note_types import NoteType, load_note_types

SHARED_DIR = "shared"
SHARED_BRANCH = "main"
RESERVED_SHARED_MARKDOWN = {
    "README.md",
    "README.markdown",
    "LICENSE.md",
    "CHANGELOG.md",
}


@dataclass(frozen=True)
class DeckSource:
    """A filesystem root that contributes decks to one logical collection."""

    root: Path
    source_id: str
    note_types_dir: Path
    is_shared: bool = False

    @classmethod
    def local(cls, collection_dir: Path, note_types_dir: Path | None = None):
        return cls(
            root=collection_dir,
            source_id="",
            note_types_dir=note_types_dir or collection_dir / NOTE_TYPES_DIR,
            is_shared=False,
        )

    @classmethod
    def shared(cls, collection_dir: Path, owner: str, repo: str):
        root = collection_dir / SHARED_DIR / owner / repo
        return cls(
            root=root,
            source_id=f"{SHARED_DIR}/{owner}/{repo}",
            note_types_dir=root / NOTE_TYPES_DIR,
            is_shared=True,
        )

    @property
    def display_name(self) -> str:
        return self.source_id or "local"

    @property
    def github_slug(self) -> str | None:
        if not self.is_shared:
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
        if not self.is_shared:
            return name
        prefix = f"{self.source_id}/"
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def unscoped_note_type_name(self, name: str) -> str:
        if not self.is_shared:
            return name
        prefix = f"{self.source_id}/"
        return name[len(prefix) :] if name.startswith(prefix) else name


@dataclass(frozen=True)
class SourceNoteTypes:
    source: DeckSource
    note_types: list[NoteType]


def is_reserved_shared_markdown(path: Path) -> bool:
    name = path.name
    return name in RESERVED_SHARED_MARKDOWN or name.startswith("_")


def deck_files_for_source(source: DeckSource) -> list[Path]:
    files = sorted(source.root.glob("*.md"))
    if not source.is_shared:
        return files
    return [path for path in files if not is_reserved_shared_markdown(path)]


def discover_deck_sources(
    collection_dir: Path,
    *,
    note_types_dir: Path | None = None,
) -> list[DeckSource]:
    sources = [DeckSource.local(collection_dir, note_types_dir)]
    shared_root = collection_dir / SHARED_DIR
    if not shared_root.exists():
        return sources

    for owner_dir in sorted(shared_root.iterdir(), key=lambda path: path.name):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir(), key=lambda path: path.name):
            if not repo_dir.is_dir():
                continue
            sources.append(
                DeckSource.shared(collection_dir, owner_dir.name, repo_dir.name)
            )
    return sources


def load_note_types_for_source(source: DeckSource) -> list[NoteType]:
    configs = load_note_types(source.note_types_dir)
    if not source.is_shared:
        return configs
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def load_note_types_for_sources(sources: list[DeckSource]) -> list[SourceNoteTypes]:
    return [
        SourceNoteTypes(source=source, note_types=load_note_types_for_source(source))
        for source in sources
    ]

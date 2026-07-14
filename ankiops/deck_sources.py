"""Deck source identity and collection source loading."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ankiops.collection import NOTE_TYPES_DIR
from ankiops.git import GitRepository
from ankiops.note_types import NoteType, load_note_types

LOCAL_SOURCE_PATH = Path(".")
COLLAB_DIR = "collab"
RESERVED_MARKDOWN_FILES = {
    "CHANGELOG.MD",
    "CODE_OF_CONDUCT.MD",
    "CONTRIBUTING.MD",
    "FUNDING.MD",
    "LICENSE.MD",
    "README.MD",
    "SECURITY.MD",
    "SUPPORT.MD",
}


def _parse_github_slug(github_slug: str) -> tuple[str, str]:
    parts = github_slug.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Expected collab source as owner/repo")
    if any("\\" in part for part in parts):
        raise ValueError("Expected collab source as owner/repo")
    return parts[0], parts[1]


@dataclass(frozen=True)
class DeckSource:
    """A canonical filesystem source for one or more decks and their assets."""

    collection_root: Path
    relative_path: Path

    def __post_init__(self) -> None:
        parts = self.relative_path.parts
        if self.relative_path == LOCAL_SOURCE_PATH:
            return
        if (
            self.relative_path.is_absolute()
            or len(parts) != 3
            or parts[0] != COLLAB_DIR
        ):
            raise ValueError("Expected source path as . or collab/owner/repo")
        _parse_github_slug("/".join(parts[1:]))

    @classmethod
    def local(cls, collection_root: Path) -> "DeckSource":
        return cls(collection_root=collection_root, relative_path=LOCAL_SOURCE_PATH)

    @classmethod
    def collab(cls, collection_root: Path, github_slug: str) -> "DeckSource":
        owner, repository = _parse_github_slug(github_slug)
        return cls(
            collection_root=collection_root,
            relative_path=Path(COLLAB_DIR) / owner / repository,
        )

    @property
    def root(self) -> Path:
        return self.collection_root / self.relative_path

    @property
    def source_path(self) -> str:
        return self.relative_path.as_posix()

    @property
    def note_types_dir(self) -> Path:
        return self.root / NOTE_TYPES_DIR

    @property
    def is_collab(self) -> bool:
        return self.relative_path != LOCAL_SOURCE_PATH

    @property
    def display_name(self) -> str:
        return self.github_slug or "local"

    @property
    def github_slug(self) -> str | None:
        if not self.is_collab:
            return None
        return "/".join(self.relative_path.parts[1:])

    @property
    def github_url(self) -> str | None:
        slug = self.github_slug
        return f"https://github.com/{slug}.git" if slug else None

    def scope_note_type_name(self, name: str) -> str:
        if not self.is_collab:
            return name
        prefix = f"{self.source_path}/"
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def unscoped_note_type_name(self, name: str) -> str:
        if not self.is_collab:
            return name
        prefix = f"{self.source_path}/"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def deck_files(self) -> list[Path]:
        files = []
        for path in sorted(self.root.glob("*.md")):
            if path.name.upper() in RESERVED_MARKDOWN_FILES:
                continue
            if "___" in path.stem:
                raise ValueError(
                    f"Ambiguous deck filename '{path.name}': do not place '_' "
                    "next to the '__' subdeck separator."
                )
            files.append(path)
        return files


def discover_deck_sources(
    collection_root: Path,
) -> list[DeckSource]:
    """Parse the canonical filesystem registry into deterministic deck sources."""
    local = DeckSource.local(collection_root)
    collab_root = collection_root / COLLAB_DIR
    if not collab_root.is_dir():
        return [local]

    collab = []
    for owner_dir in sorted(collab_root.iterdir(), key=lambda path: path.name):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir(), key=lambda path: path.name):
            if not repo_dir.is_dir():
                continue
            source = DeckSource.collab(
                collection_root, f"{owner_dir.name}/{repo_dir.name}"
            )
            if not GitRepository(source.root).is_repo():
                raise ValueError(
                    f"Collab source directory {source.root} is not an independent "
                    "Git repository. Leave it untouched for inspection and subscribe "
                    f"to {source.github_slug} again in a fresh collection path."
                )
            collab.append(source)
    return [local, *collab]


def load_note_types_for_source(source: DeckSource) -> list[NoteType]:
    configs = load_note_types(source.note_types_dir)
    if not source.is_collab:
        return configs
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def load_note_types_for_collection(
    collection_root: Path,
) -> list[NoteType]:
    """Load note types from every source in the collection filesystem."""
    note_types = []
    for source in discover_deck_sources(collection_root):
        note_types.extend(load_note_types_for_source(source))
    return note_types

"""Deck source identity and collection source loading."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from blake3 import blake3

from ankiops.collection import NOTE_TYPES_DIR
from ankiops.note_types import NoteType, load_note_types

LOCAL_SOURCE_ID = "local"
SHARED_DIR = "shared"
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


def _validate_shared_source_id(source_id: str) -> tuple[str, str]:
    parts = source_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Expected shared source as owner/repo")
    if any("\\" in part for part in parts):
        raise ValueError("Expected shared source as owner/repo")
    return parts[0], parts[1]


@dataclass(frozen=True)
class DeckSource:
    """One filesystem source participating in a logical collection."""

    collection_dir: Path
    source_id: str

    @classmethod
    def local(cls, collection_dir: Path) -> "DeckSource":
        return cls(collection_dir=collection_dir, source_id=LOCAL_SOURCE_ID)

    @classmethod
    def shared(cls, collection_dir: Path, source_id: str) -> "DeckSource":
        _validate_shared_source_id(source_id)
        return cls(collection_dir=collection_dir, source_id=source_id)

    @property
    def root(self) -> Path:
        if not self.is_shared:
            return self.collection_dir
        owner, repo = _validate_shared_source_id(self.source_id)
        return self.collection_dir / SHARED_DIR / owner / repo

    @property
    def note_types_dir(self) -> Path:
        return self.root / NOTE_TYPES_DIR

    @property
    def is_shared(self) -> bool:
        return self.source_id != LOCAL_SOURCE_ID

    @property
    def display_name(self) -> str:
        return self.source_id

    @property
    def github_slug(self) -> str | None:
        return self.source_id if self.is_shared else None

    @property
    def github_url(self) -> str | None:
        slug = self.github_slug
        return f"https://github.com/{slug}.git" if slug else None

    def scope_note_type_name(self, name: str) -> str:
        if not self.is_shared:
            return name
        prefix = f"{SHARED_DIR}/{self.source_id}/"
        return name if name.startswith(prefix) else f"{prefix}{name}"

    def unscoped_note_type_name(self, name: str) -> str:
        if not self.is_shared:
            return name
        prefix = f"{SHARED_DIR}/{self.source_id}/"
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
    collection_dir: Path,
    *,
    note_types_dir: Path | None = None,
) -> list[DeckSource]:
    """Discover valid nested repositories in the reserved shared directory."""
    local = DeckSource.local(collection_dir)
    shared_root = collection_dir / SHARED_DIR
    if not shared_root.is_dir():
        return [local]

    shared = []
    for owner_dir in sorted(shared_root.iterdir(), key=lambda path: path.name):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir(), key=lambda path: path.name):
            if not repo_dir.is_dir():
                continue
            shared.append(
                DeckSource.shared(collection_dir, f"{owner_dir.name}/{repo_dir.name}")
            )
    return [local, *shared]


def load_note_types_for_source(source: DeckSource) -> list[NoteType]:
    configs = load_note_types(source.note_types_dir)
    if not source.is_shared:
        return configs
    return [
        replace(config, name=source.scope_note_type_name(config.name))
        for config in configs
    ]


def load_note_types_for_collection(
    collection_dir: Path,
    *,
    sources: Sequence[DeckSource] | None = None,
    note_types_dir: Path | None = None,
) -> list[NoteType]:
    """Load note types from the explicitly selected collection sources."""
    selected = (
        list(sources) if sources is not None else discover_deck_sources(collection_dir)
    )
    note_types = []
    for source in selected:
        if not source.is_shared and note_types_dir is not None:
            note_types.extend(load_note_types(note_types_dir))
        else:
            note_types.extend(load_note_types_for_source(source))
    return note_types


def source_content_hash(source: DeckSource) -> str:
    """Hash the visible source tree without reading repository metadata."""
    digest = blake3()
    for path in sorted(source.root.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(source.root).parts:
            continue
        relative = path.relative_to(source.root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    return digest.hexdigest()

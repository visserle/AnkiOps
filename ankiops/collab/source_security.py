"""Trust-boundary checks for subscribed collab repositories."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from ankiops.deck_sources import RESERVED_MARKDOWN_FILES, discover_deck_sources
from ankiops.git import GitRepository
from ankiops.note_types import load_note_types


def _is_anki_source_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if (
        len(candidate.parts) == 1
        and candidate.suffix.lower() == ".md"
        and candidate.name.upper() not in RESERVED_MARKDOWN_FILES
    ):
        return True
    return bool(candidate.parts and candidate.parts[0] == "note_types")


def _ensure_regular_without_symlinks(
    root: Path,
    path: Path,
    *,
    display_name: str,
) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"The subscribed deck {display_name} contains a symbolic link at "
                f"'{path.relative_to(root)}'. Deck Markdown and note-type files "
                "must be regular files contained in the subscribed repository."
            )
    if path.exists() and not path.is_file():
        raise ValueError(
            f"The subscribed deck {display_name} contains a non-regular file at "
            f"'{path.relative_to(root)}'. Deck Markdown and note-type files must "
            "be regular files."
        )


def validate_collab_worktree(
    repository: GitRepository,
    *,
    display_name: str,
) -> None:
    """Reject unsafe applicable paths in the checked-out working tree."""
    root = repository.root
    for path in root.iterdir():
        candidate = PurePosixPath(path.name)
        if (
            candidate.suffix.lower() == ".md"
            and candidate.name.upper() not in RESERVED_MARKDOWN_FILES
        ):
            _ensure_regular_without_symlinks(root, path, display_name=display_name)

    note_types_dir = root / "note_types"
    if note_types_dir.is_symlink():
        _ensure_regular_without_symlinks(
            root, note_types_dir, display_name=display_name
        )
    if not note_types_dir.is_dir():
        return
    for path in note_types_dir.rglob("*"):
        is_special = path.exists() and not path.is_file() and not path.is_dir()
        if path.is_symlink() or is_special:
            _ensure_regular_without_symlinks(root, path, display_name=display_name)

    load_note_types(note_types_dir)


def validate_collab_checkout(
    repository: GitRepository,
    *,
    display_name: str,
    ref: str = "HEAD",
) -> None:
    """Reject Git symlinks in files that AnkiOps can parse or apply."""
    output = repository.run(["ls-tree", "-r", "-z", ref]).stdout
    for raw_entry in output.split("\0"):
        if not raw_entry:
            continue
        metadata, path = raw_entry.split("\t", 1)
        mode = metadata.split(" ", 1)[0]
        if _is_anki_source_path(path) and mode not in {"100644", "100755"}:
            kind = "symbolic link" if mode == "120000" else "non-regular Git entry"
            raise ValueError(
                f"The subscribed deck {display_name} contains a {kind} at '{path}'. "
                "Deck Markdown and note-type files must be regular files contained "
                "in the subscribed repository."
            )


def validate_collection_collab_sources(collection_root: Path) -> None:
    """Validate all subscribed checkouts before a collection sync reads them."""
    for source in discover_deck_sources(collection_root)[1:]:
        repository = GitRepository(source.root)
        validate_collab_checkout(repository, display_name=source.display_name)
        validate_collab_worktree(repository, display_name=source.display_name)

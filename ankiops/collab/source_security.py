"""Trust-boundary checks for subscribed collab repositories."""

from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ankiops.deck_sources import RESERVED_MARKDOWN_FILES, discover_deck_sources
from ankiops.git import GitRepository
from ankiops.note_types import load_note_types


@dataclass(frozen=True)
class ProtectedWorktreePaths:
    untracked: frozenset[str]
    ignored: frozenset[str]


def _validate_conflict_resolution_location(
    conflict_root: Path,
    resolution: Path,
) -> None:
    lexical_root = conflict_root.absolute()
    lexical_resolution = resolution.absolute()
    try:
        relative = lexical_resolution.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(
            f"Conflict resolution '{resolution}' escapes its conflict directory."
        ) from error

    current = lexical_root
    if current.is_symlink() or not current.is_dir():
        raise ValueError(
            f"Conflict resolution '{resolution}' must be a regular non-symlink "
            "file contained in its conflict directory."
        )
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"Conflict resolution '{resolution}' must be a regular "
                "non-symlink file contained in its conflict directory."
            )


def read_regular_conflict_resolution(
    conflict_root: Path,
    resolution: Path,
) -> bytes | None:
    """Read a contained regular resolution without following its final symlink."""
    _validate_conflict_resolution_location(conflict_root, resolution)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(resolution, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(
            f"Conflict resolution '{resolution}' must be a regular non-symlink "
            "file contained in its conflict directory."
        ) from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(
            f"Conflict resolution '{resolution}' must be a regular non-symlink "
            "file contained in its conflict directory."
        )
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _tree_paths(repository: GitRepository, ref: str) -> set[str]:
    output = repository.run(["ls-tree", "-r", "--name-only", "-z", ref]).stdout
    return {path for path in output.split("\0") if path}


def protected_worktree_paths(repository: GitRepository) -> ProtectedWorktreePaths:
    """Return untracked and ignored files owned by the local user."""
    untracked = repository.run(
        ["ls-files", "--others", "--exclude-standard", "-z"]
    ).stdout
    ignored = repository.run(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
    ).stdout
    return ProtectedWorktreePaths(
        untracked=frozenset(path for path in untracked.split("\0") if path),
        ignored=frozenset(path for path in ignored.split("\0") if path),
    )


def _paths_collide(left: str, right: str) -> bool:
    left_parts = tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in PurePosixPath(left).parts
    )
    right_parts = tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in PurePosixPath(right).parts
    )
    shared = min(len(left_parts), len(right_parts))
    return left_parts[:shared] == right_parts[:shared]


def _protected_file_matches_candidate(
    current_repository: GitRepository,
    candidate_repository: GitRepository,
    *,
    protected_path: str,
    candidate_path: str,
    candidate_ref: str,
) -> bool:
    if protected_path != candidate_path:
        return False

    local_path = current_repository.root
    for part in PurePosixPath(protected_path).parts:
        local_path /= part
        if local_path.is_symlink():
            return False
    try:
        local_mode = local_path.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(local_mode):
            return False
    except OSError:
        return False

    entry = candidate_repository.run(
        ["ls-tree", "-z", candidate_ref, "--", candidate_path]
    ).stdout.rstrip("\0")
    if not entry:
        return False
    metadata, tree_path = entry.split("\t", 1)
    mode, kind, object_id = metadata.split(" ", 2)
    if (
        tree_path != candidate_path
        or kind != "blob"
        or mode not in {"100644", "100755"}
        or (mode == "100755") != bool(local_mode & 0o111)
    ):
        return False

    local_object = current_repository.run(
        ["hash-object", "--no-filters", "--", protected_path], check=False
    )
    return local_object.returncode == 0 and local_object.stdout.strip() == object_id


def validate_candidate_preserves_protected_paths(
    current_repository: GitRepository,
    candidate_repository: GitRepository,
    *,
    current_ref: str,
    candidate_ref: str,
    protected_paths: ProtectedWorktreePaths,
    display_name: str,
) -> None:
    """Reject upstream paths that would replace local untracked or ignored files."""
    current_paths = _tree_paths(current_repository, current_ref)
    new_candidate_paths = (
        _tree_paths(candidate_repository, candidate_ref) - current_paths
    )
    all_protected_paths = protected_paths.untracked | protected_paths.ignored
    for protected_path in sorted(all_protected_paths):
        for candidate_path in sorted(new_candidate_paths):
            if not _paths_collide(protected_path, candidate_path):
                continue
            if (
                protected_path not in protected_paths.ignored
                and _protected_file_matches_candidate(
                    current_repository,
                    candidate_repository,
                    protected_path=protected_path,
                    candidate_path=candidate_path,
                    candidate_ref=candidate_ref,
                )
            ):
                continue
            raise ValueError(
                f"The update for {display_name} would overwrite or remove local "
                f"untracked or ignored file '{protected_path}' because upstream "
                f"now tracks '{candidate_path}'. The subscribed files were left "
                "untouched. Move or remove the local file, then retry."
            )


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

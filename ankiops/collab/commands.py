"""Independent-repository collab source commands."""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from ankiops.anki_manifest import (
    AnkiApplicableManifest,
    anki_applicable_paths_changed,
    source_anki_manifest,
)
from ankiops.collab.errors import (
    RepositoryCollisionError,
    format_missing_note_keys_error,
)
from ankiops.collab.hosting import GitHubHost, PullRequestInfo
from ankiops.collab.publish import publish_collab_deck
from ankiops.collab.source_security import (
    protected_worktree_paths,
    read_regular_conflict_resolution,
    validate_candidate_preserves_protected_paths,
    validate_collab_checkout,
    validate_collab_worktree,
)
from ankiops.collection import require_collection_root
from ankiops.console import print_line as output_line
from ankiops.console import print_next_steps as output_next_steps
from ankiops.console import print_result as output_result
from ankiops.deck_sources import (
    RESERVED_MARKDOWN_FILES,
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
)
from ankiops.git import GitPathChange, GitRepository
from ankiops.markdown import read_deck_file
from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)

JOURNAL_BRANCH = "ankiops/journal"
INTEGRATED_REF = "refs/ankiops/integrated"
UPLOADED_REF = "refs/ankiops/uploaded"
_SAFE_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$")
_SAFE_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CLEAN_MERGE_CONFIRMATION = "CONFIRM_CLEAN_MERGE"
_CLEAN_MERGE_REFRESH = ".clean-merge-refresh"
_UNRESOLVED_CONFLICT_PLACEHOLDER = (
    b"<<<<<<< AnkiOps unresolved conflict\n"
    b"Git could not create a textual merge for this path. Raw versions are stored "
    b"beside\nthis file as .base, .local, and .upstream when that version exists.\n"
    b"=======\n"
    b"Replace this entire placeholder with the desired final contents, or delete "
    b"this file\nto resolve the conflict by deleting the path, before retrying.\n"
    b">>>>>>> AnkiOps unresolved conflict\n"
)


class _ConflictResolutionRequired(ValueError):
    pass


class _RepositoryRelation(Enum):
    CURRENT = auto()
    AHEAD = auto()
    BEHIND = auto()
    DIVERGED = auto()


@dataclass(frozen=True)
class _RepositoryState:
    relation: _RepositoryRelation
    local_commits: int
    upstream_commits: int


def _parse_github_slug(repository: str) -> str:
    value = repository.strip().removesuffix(".git")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Expected GitHub repository as owner/repo")
    owner, repo = parts
    if (
        not _SAFE_OWNER_RE.fullmatch(owner)
        or not _SAFE_REPOSITORY_RE.fullmatch(repo)
        or repo in {".", ".."}
    ):
        raise ValueError(
            f"Invalid collab deck identity '{repository}': owner may use ASCII "
            "letters, digits, hyphens, and internal underscores; repository may "
            "use ASCII letters, digits, hyphens, underscores, and periods."
        )
    return value


def _collab_source(collection_root: Path, repository: str) -> DeckSource:
    return DeckSource.collab(collection_root, _parse_github_slug(repository))


def _require_collection_git(collection_root: Path) -> GitRepository:
    collection_git = GitRepository(collection_root)
    collection_git.ensure_repo(
        "AnkiOps collections require a Git repository at the root."
    )
    ignored = collection_git.run(
        ["check-ignore", "-q", "--no-index", "collab/.ankiops-probe"],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "This collection is not configured to keep collab decks separate. "
            "Add /collab/ to the root .gitignore, then retry the command."
        )
    return collection_git


def _require_source_git(source: DeckSource) -> GitRepository:
    source_git = GitRepository(source.root)
    source_git.ensure_repo(
        f"The subscribed deck {source.display_name} is not a valid independent "
        f"repository at {source.root}. Leave the directory untouched for "
        "inspection and subscribe in a fresh collection path."
    )
    if source_git.remote_url("upstream") is None:
        raise ValueError(
            f"The subscribed deck {source.display_name} has no GitHub source. "
            f"Subscribe to it again in a fresh collection path: ankiops collab "
            f"subscribe {source.display_name}"
        )
    return source_git


def _ensure_submittable_note_keys(source: DeckSource) -> None:
    configs = load_note_types_for_source(source)
    missing = []
    note_key_locations: dict[str, list[str]] = {}
    for md_file in source.deck_files():
        try:
            parsed = read_deck_file(
                md_file, note_types=configs, context_root=source.root
            )
        except ValueError as error:
            identity = re.search(
                r"Unknown note type 'collab/([^/']+)/([^/']+)/",
                str(error),
            )
            canonical = None
            if identity:
                try:
                    canonical = _parse_github_slug("/".join(identity.groups()))
                except ValueError:
                    pass
            if canonical and canonical != source.display_name:
                raise ValueError(
                    f"This repository's deck files belong to {canonical}, not "
                    f"{source.display_name}. Subscribe to the canonical deck "
                    f"instead: ankiops collab subscribe {canonical}"
                ) from error
            raise
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing.append(f"{md_file.name} note {index}")
            else:
                note_key_locations.setdefault(note.note_key, []).append(
                    f"{md_file.name} note {index}"
                )
    if missing:
        raise ValueError(format_missing_note_keys_error(len(missing)))
    duplicates = [
        f"Duplicate note_key '{note_key}' in {', '.join(locations)}"
        for note_key, locations in note_key_locations.items()
        if len(locations) > 1
    ]
    if duplicates:
        raise ValueError("; ".join(duplicates))


def _log_result(
    action: str,
    source: DeckSource,
    summary: str,
    *,
    details: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> None:
    output_result(action, source.display_name, summary)
    for detail in details or []:
        output_line(f"  {detail}")
    _log_next_steps(next_steps or [])


def _log_next_steps(steps: list[str]) -> None:
    output_next_steps(steps)


def _read_anki_manifest(
    source: DeckSource,
) -> AnkiApplicableManifest | None:
    try:
        return source_anki_manifest(source)
    except Exception as error:
        logger.debug(
            "Could not calculate Anki-applicable paths for %s: %s",
            source.display_name,
            error,
        )
        return None


def _conservative_anki_guidance(changed_paths: set[str]) -> bool:
    for raw_path in changed_paths:
        path = Path(raw_path)
        if (
            len(path.parts) == 1
            and path.suffix.lower() == ".md"
            and path.name.upper() not in RESERVED_MARKDOWN_FILES
        ):
            return True
        if path.parts and path.parts[0] in {"media", "note_types"}:
            return True
    return False


def _derive_submit_title(source_git: GitRepository, upstream_ref: str) -> str:
    changes = source_git.diff_name_status(upstream_ref, "HEAD")
    deck_changes: list[tuple[str, tuple[Path, ...]]] = []
    assets_changed = False
    for change in changes:
        deck_paths = []
        for raw_path in change.paths:
            path = Path(raw_path)
            if (
                len(path.parts) == 1
                and path.suffix.lower() == ".md"
                and path.name.upper() not in RESERVED_MARKDOWN_FILES
            ):
                deck_paths.append(path)
            elif path.parts and path.parts[0] in {"media", "note_types"}:
                assets_changed = True
        if deck_paths:
            deck_changes.append((change.status[:1], tuple(deck_paths)))
    if len(deck_changes) == 1:
        status, paths = deck_changes[0]
        if status == "R" and len(paths) == 2:
            old_name = paths[0].stem.replace("__", " › ")
            new_name = paths[1].stem.replace("__", " › ")
            return f"Rename {old_name} to {new_name}"
        name = paths[-1].stem.replace("__", " › ")
        verb = {"A": "Add", "D": "Remove"}.get(status, "Update")
        return f"{verb} {name}"
    if len(deck_changes) > 1:
        return f"Update {len(deck_changes)} decks"
    if assets_changed:
        return "Update shared deck assets"
    return "Update deck documentation"


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _source_operation(
    state: SyncState, source: DeckSource, kind: str
) -> tuple[str, dict[str, str | None] | None]:
    existing = state.get_collab_operation(source.source_path)
    if existing and existing["kind"] != kind:
        raise ValueError(
            f"{source.display_name} has an unfinished {existing['kind']} action. "
            f"Run ankiops collab status {source.display_name} for the exact next step."
        )
    operation_id = str(existing["operation_id"]) if existing else uuid4().hex[:12]
    return operation_id, existing


def _conflict_directory(
    collection_root: Path, source: DeckSource, operation_id: str
) -> Path:
    owner, repo_name = source.display_name.split("/", 1)
    return collection_root / ".ankiops" / "conflicts" / owner / repo_name / operation_id


def _read_repository_version(
    repository: GitRepository, ref: str, rel_path: str
) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=repository.root,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _save_conflict_copies(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
    operation_id: str,
) -> Path:
    conflict_root = _conflict_directory(collection_root, source, operation_id)
    for rel_path in source_git.unmerged_paths():
        versions: dict[str, bytes] = {}
        for stage, label in ((1, "base"), (2, "local"), (3, "upstream")):
            content = _read_repository_version(source_git, f":{stage}", rel_path)
            if content is None:
                continue
            versions[label] = content
            target = conflict_root / f"{rel_path}.{label}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        editable = conflict_root / rel_path
        editable.parent.mkdir(parents=True, exist_ok=True)
        merged = source_git.root / rel_path
        if merged.exists():
            editable_content = merged.read_bytes()
        else:
            editable_content = versions.get("local", versions.get("upstream", b""))
        if _contains_conflict_markers(editable_content):
            editable.write_bytes(editable_content)
        else:
            editable.write_bytes(_UNRESOLVED_CONFLICT_PLACEHOLDER)
    return conflict_root


def _preserved_conflict_paths(conflict_root: Path) -> list[str]:
    paths = set()
    for artifact in conflict_root.rglob("*"):
        if not artifact.is_file():
            continue
        relative = artifact.relative_to(conflict_root).as_posix()
        if relative.startswith("prior/"):
            continue
        for suffix in (".base", ".local", ".upstream"):
            if relative.endswith(suffix):
                paths.add(relative[: -len(suffix)])
                break
    return sorted(paths)


def _save_clean_merge_confirmation(
    collection_root: Path,
    source: DeckSource,
    repository: GitRepository,
    operation_id: str,
    previous_root: Path,
    *,
    base_tree: str,
    local_tree: str,
    upstream_tree: str,
    candidate_tree: str,
) -> Path:
    conflict_root = _conflict_directory(
        collection_root,
        source,
        f"{operation_id}-refresh-{uuid4().hex[:6]}",
    )
    conflict_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(previous_root, conflict_root / "prior")
    paths = _preserved_conflict_paths(previous_root)
    for rel_path in paths:
        previous = previous_root / rel_path
        if not previous.exists():
            previous = previous_root / f"{rel_path}.previous"
        if previous.exists():
            target = conflict_root / f"{rel_path}.previous"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(previous.read_bytes())
        else:
            target = conflict_root / f"{rel_path}.previous-deleted"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("The previous resolution deleted this path.\n")
        for label, tree in (
            ("base", base_tree),
            ("local", local_tree),
            ("upstream", upstream_tree),
            ("candidate", candidate_tree),
        ):
            content = _read_repository_version(repository, tree, rel_path)
            if content is None:
                continue
            target = conflict_root / f"{rel_path}.{label}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    (conflict_root / _CLEAN_MERGE_CONFIRMATION).write_text(
        "Upstream changed after conflict resolution began, and the new merge is "
        "clean.\nReview each path's .previous and .candidate versions, plus its "
        ".base, .local, and .upstream evidence when present. Delete only this "
        "confirmation file to accept the clean candidate on retry.\n",
        encoding="utf-8",
    )
    (conflict_root / _CLEAN_MERGE_REFRESH).write_text(
        f"previous={previous_root}\nupstream_tree={upstream_tree}\n",
        encoding="utf-8",
    )
    return conflict_root


def _upstream_ref(source_git: GitRepository) -> tuple[str, str]:
    default_branch = source_git.default_branch("upstream")
    return default_branch, f"upstream/{default_branch}"


def _repository_fingerprint(source_git: GitRepository) -> str:
    digest = hashlib.sha256()
    for args in (
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        ["diff", "--binary"],
        ["diff", "--cached", "--binary"],
    ):
        digest.update(source_git.run(args).stdout.encode())
    for path in sorted(
        path
        for path in source_git.root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ):
        digest.update(str(path.relative_to(source_git.root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_repository_files(source: Path, target: Path) -> None:
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in source.iterdir():
        if child.name == ".git":
            continue
        destination = target / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination, symlinks=True)
        elif child.is_symlink():
            destination.symlink_to(child.readlink())
        else:
            shutil.copy2(child, destination)


def _copy_git_identity(source_git: GitRepository, target_git: GitRepository) -> None:
    for key in ("user.name", "user.email"):
        value = source_git.run(["config", "--get", key], check=False).stdout.strip()
        if value:
            target_git.run(["config", key, value])


def _contains_conflict_markers(content: bytes) -> bool:
    return all(marker in content for marker in (b"<<<<<<<", b"=======", b">>>>>>>"))


def _has_conflict_markers(path: Path) -> bool:
    return path.exists() and _contains_conflict_markers(path.read_bytes())


def _apply_preserved_resolutions(
    transaction_git: GitRepository, conflict_root: Path
) -> list[str]:
    unresolved = []
    for rel_path in transaction_git.unmerged_paths():
        resolution = conflict_root / rel_path
        resolution_content = read_regular_conflict_resolution(conflict_root, resolution)
        if resolution_content is None:
            transaction_git.run(["rm", "-f", "--", rel_path])
            continue
        if _contains_conflict_markers(resolution_content):
            unresolved.append(rel_path)
            continue
        target = transaction_git.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resolution_content)
        transaction_git.run(["add", "--", rel_path])
    return unresolved


def _merge_content_trees(
    repository: GitRepository,
    *,
    base_tree: str,
    local_tree: str,
    upstream_tree: str,
    message: str,
) -> str:
    """Merge two content trees using the supplied tree as their common base."""
    if local_tree == upstream_tree or upstream_tree == base_tree:
        return local_tree
    if local_tree == base_tree:
        upstream_commit = repository.create_commit(
            upstream_tree, repository.head(), message
        )
        repository.reset_hard(upstream_commit)
        return upstream_tree

    base_commit = repository.create_commit(base_tree, None, "AnkiOps merge base")
    local_commit = repository.create_commit(
        local_tree, base_commit, "AnkiOps local draft"
    )
    upstream_commit = repository.create_commit(
        upstream_tree, base_commit, "AnkiOps upstream content"
    )
    repository.reset_hard(local_commit)
    repository.integrate(upstream_commit, message)
    result = repository.tree("HEAD")
    if result is None:
        raise ValueError("Could not read the integrated content tree.")
    return result


def _tree_delta_is_present(
    repository: GitRepository,
    *,
    base_tree: str,
    changed_tree: str,
    candidate_tree: str,
) -> bool:
    """Return whether every path changed from base has that value in candidate."""
    paths = repository.diff_paths(base_tree, changed_tree)
    if not paths:
        return False
    for path in paths:
        result = repository.run(
            ["diff", "--quiet", changed_tree, candidate_tree, "--", path],
            check=False,
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        if result.returncode == 1:
            return False
    return True


def _integrate_upstream(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
    state: SyncState,
    *,
    kind: str,
    uploaded_accepted: bool = False,
    requested_title: str | None = None,
) -> tuple[str, bool, str | None, str, str]:
    operation_id, existing = _source_operation(state, source, kind)
    original_commit = source_git.head()
    if original_commit is None:
        raise ValueError(f"Collab source {source.display_name} has no commits.")
    validate_collab_checkout(source_git, display_name=source.display_name)
    validate_collab_worktree(source_git, display_name=source.display_name)
    protected_paths = protected_worktree_paths(source_git)
    original_fingerprint = _repository_fingerprint(source_git)

    if existing and existing["state"] == "applying":
        prepared_head = existing.get("prepared_head")
        upstream_tree = existing.get("upstream_tree")
        expected_head = existing.get("expected_head")
        if (
            prepared_head
            and upstream_tree
            and original_commit == prepared_head
            and not source_git.status_lines()
        ):
            source_git.fetch("upstream")
            default_branch, upstream_ref = _upstream_ref(source_git)
            integrated_commit = next(
                (
                    commit
                    for commit in source_git.run(
                        ["rev-list", upstream_ref]
                    ).stdout.splitlines()
                    if source_git.tree(commit) == upstream_tree
                ),
                None,
            )
            if integrated_commit is None:
                raise ValueError(
                    f"Could not repair the interrupted update for "
                    f"{source.display_name}: its prepared upstream version is no "
                    "longer available. The subscribed files were left untouched."
                )
            source_git.update_ref(INTEGRATED_REF, integrated_commit)
            if existing.get("recovery_ref"):
                shutil.rmtree(str(existing["recovery_ref"]), ignore_errors=True)
            return (
                operation_id,
                prepared_head != expected_head,
                None,
                str(upstream_tree),
                default_branch,
            )

    if existing and existing["state"] in {"integrating", "conflict", "applying"}:
        if (
            existing.get("expected_head") != original_commit
            or existing.get("expected_fingerprint") != original_fingerprint
        ):
            raise ValueError(
                f"The subscribed deck changed after its interrupted update. Its "
                f"files were left untouched. Review ankiops collab status "
                f"{source.display_name} before retrying."
            )
    else:
        existing = None

    operation_values = {}
    if kind == "submit" and requested_title is not None:
        operation_values["requested_title"] = requested_title
    state.save_collab_operation(
        source.source_path,
        operation_id,
        kind,
        "integrating",
        expected_head=original_commit,
        expected_fingerprint=original_fingerprint,
        prepared_head=None,
        upstream_tree=None,
        recovery_ref=str(existing.get("recovery_ref") or "") if existing else None,
        last_error=None,
        **operation_values,
    )
    transaction_parent = collection_root / ".ankiops" / "transactions"
    transaction_parent.mkdir(parents=True, exist_ok=True)
    transaction_path = (
        Path(tempfile.mkdtemp(prefix=f"{operation_id}-", dir=transaction_parent))
        / "deck"
    )
    conflict_root: Path | None = None
    try:
        transaction_git = GitRepository.clone(
            str(source.root), transaction_path, remote="local"
        )
        _copy_git_identity(source_git, transaction_git)
        _copy_repository_files(source.root, transaction_git.root)
        saved_commit = transaction_git.checkpoint(
            f"Save local deck changes for {source.display_name}"
        )
        local_journal_commit = transaction_git.head()
        local_tree = transaction_git.tree("HEAD")
        if local_journal_commit is None or local_tree is None:
            raise ValueError(
                f"Could not read the local draft for {source.display_name}."
            )
        upstream_url = source_git.remote_url("upstream")
        if upstream_url is None:
            raise ValueError(
                f"No GitHub source is configured for {source.display_name}."
            )
        transaction_git.run(["remote", "add", "upstream", upstream_url])
        transaction_git.fetch("upstream")
        default_branch, upstream_ref = _upstream_ref(transaction_git)
        upstream_tree = transaction_git.tree(upstream_ref)
        if upstream_tree is None:
            raise ValueError(
                f"Could not read the available update for {source.display_name}."
            )
        validate_collab_checkout(
            transaction_git,
            display_name=source.display_name,
            ref=upstream_ref,
        )
        validate_candidate_preserves_protected_paths(
            source_git,
            transaction_git,
            current_ref=original_commit,
            candidate_ref=upstream_ref,
            protected_paths=protected_paths,
            display_name=source.display_name,
        )
        if (
            existing
            and existing["state"] == "conflict"
            and existing.get("recovery_ref")
        ):
            recovery_root = Path(str(existing["recovery_ref"]))
            is_clean_refresh = (recovery_root / _CLEAN_MERGE_REFRESH).exists()
            if is_clean_refresh and existing.get("upstream_tree") == upstream_tree:
                if (recovery_root / _CLEAN_MERGE_CONFIRMATION).exists():
                    state.save_collab_operation(
                        source.source_path,
                        operation_id,
                        kind,
                        "conflict",
                        expected_head=original_commit,
                        expected_fingerprint=original_fingerprint,
                        prepared_head=None,
                        upstream_tree=upstream_tree,
                        recovery_ref=str(recovery_root),
                        last_error="Clean merge confirmation is still required.",
                    )
                    raise _ConflictResolutionRequired(
                        f"The upstream changed after conflict resolution began for "
                        f"{source.display_name}. The new merge is clean, but "
                        "confirmation is still required. Review the evidence in "
                        f"{recovery_root}, delete {_CLEAN_MERGE_CONFIRMATION} there "
                        "to accept it, then retry: "
                        f"ankiops collab update {source.display_name}"
                    )
                conflict_root = recovery_root
        integrated_tree = source_git.tree(INTEGRATED_REF)
        if integrated_tree is None:
            raise ValueError(
                f"Could not read the integrated checkpoint for {source.display_name}."
            )
        uploaded_tree = source_git.tree(UPLOADED_REF) if kind == "submit" else None
        uploaded_is_upstream = uploaded_accepted or bool(
            uploaded_tree
            and _tree_delta_is_present(
                transaction_git,
                base_tree=integrated_tree,
                changed_tree=uploaded_tree,
                candidate_tree=upstream_tree,
            )
        )
        base_tree = uploaded_tree if uploaded_is_upstream else integrated_tree
        result_tree = _merge_content_trees(
            transaction_git,
            base_tree=base_tree,
            local_tree=local_tree,
            upstream_tree=upstream_tree,
            message=f"Integrate upstream changes for {source.display_name}",
        )
        if (
            existing
            and existing["state"] == "conflict"
            and existing.get("recovery_ref")
            and existing.get("upstream_tree")
            and existing.get("upstream_tree") != upstream_tree
        ):
            previous_root = Path(str(existing["recovery_ref"]))
            refreshed_conflict_root = _save_clean_merge_confirmation(
                collection_root,
                source,
                transaction_git,
                operation_id,
                previous_root,
                base_tree=base_tree,
                local_tree=local_tree,
                upstream_tree=upstream_tree,
                candidate_tree=result_tree,
            )
            state.save_collab_operation(
                source.source_path,
                operation_id,
                kind,
                "conflict",
                expected_head=original_commit,
                expected_fingerprint=original_fingerprint,
                prepared_head=None,
                upstream_tree=upstream_tree,
                recovery_ref=str(refreshed_conflict_root),
                last_error="Upstream advanced and the refreshed merge became clean.",
            )
            raise _ConflictResolutionRequired(
                f"The upstream advanced after the conflict for "
                f"{source.display_name} was preserved. The prior resolution was "
                "not applied, even though the refreshed merge is now clean. Review "
                f"the old and new evidence at {refreshed_conflict_root}, delete "
                f"{_CLEAN_MERGE_CONFIRMATION} there to accept the clean candidate, "
                f"then retry: ankiops collab update {source.display_name}"
            )
        original_tree = source_git.tree(original_commit)
        files_changed = original_tree != result_tree
    except subprocess.CalledProcessError as error:
        if "transaction_git" in locals() and transaction_git.unmerged_paths():
            if existing and existing.get("recovery_ref"):
                previous_upstream_tree = existing.get("upstream_tree")
                if previous_upstream_tree and previous_upstream_tree != upstream_tree:
                    refreshed_conflict_root = _save_conflict_copies(
                        collection_root,
                        source,
                        transaction_git,
                        f"{operation_id}-refresh-{uuid4().hex[:6]}",
                    )
                    state.save_collab_operation(
                        source.source_path,
                        operation_id,
                        kind,
                        "conflict",
                        expected_head=original_commit,
                        expected_fingerprint=original_fingerprint,
                        prepared_head=None,
                        upstream_tree=upstream_tree,
                        recovery_ref=str(refreshed_conflict_root),
                        last_error="Upstream advanced after conflict resolution began.",
                    )
                    shutil.rmtree(transaction_path.parent, ignore_errors=True)
                    raise ValueError(
                        f"The upstream advanced after the conflict for "
                        f"{source.display_name} was preserved. The prior resolution "
                        "was not applied and remains available for reference. Resolve "
                        f"the refreshed conflict at {refreshed_conflict_root}, then "
                        f"retry: ankiops collab update {source.display_name}"
                    )
                conflict_root = Path(str(existing["recovery_ref"]))
                try:
                    unresolved = _apply_preserved_resolutions(
                        transaction_git, conflict_root
                    )
                except BaseException as error:
                    state.save_collab_operation(
                        source.source_path,
                        operation_id,
                        kind,
                        "conflict",
                        expected_head=original_commit,
                        expected_fingerprint=original_fingerprint,
                        prepared_head=None,
                        upstream_tree=upstream_tree,
                        recovery_ref=str(conflict_root),
                        last_error=str(error),
                    )
                    shutil.rmtree(transaction_path.parent, ignore_errors=True)
                    raise
                if not unresolved:
                    transaction_git.run(["commit", "--no-edit"])
                    files_changed = True
                    default_branch, upstream_ref = _upstream_ref(transaction_git)
                    upstream_tree = transaction_git.tree(upstream_ref)
                    if upstream_tree is None:
                        raise ValueError(
                            "Could not read the available update for "
                            f"{source.display_name}."
                        )
                else:
                    conflict_root = Path(str(existing["recovery_ref"]))
            else:
                conflict_root = _save_conflict_copies(
                    collection_root, source, transaction_git, operation_id
                )
            if transaction_git.unmerged_paths():
                paths = ", ".join(transaction_git.unmerged_paths())
                state.save_collab_operation(
                    source.source_path,
                    operation_id,
                    kind,
                    "conflict",
                    expected_head=original_commit,
                    expected_fingerprint=original_fingerprint,
                    prepared_head=None,
                    upstream_tree=upstream_tree,
                    recovery_ref=str(conflict_root),
                    last_error=str(error),
                )
                shutil.rmtree(transaction_path.parent, ignore_errors=True)
                raise ValueError(
                    f"Local and upstream edits overlap in: {paths}. The subscribed "
                    f"deck was not changed. Resolve each editable file in "
                    f"{conflict_root}: replace its conflict placeholder with the "
                    "final contents, or delete it to delete that path. Then retry: "
                    f"ankiops collab update {source.display_name}"
                ) from error
        else:
            state.save_collab_operation(
                source.source_path,
                operation_id,
                kind,
                "integrating",
                expected_head=original_commit,
                expected_fingerprint=original_fingerprint,
                prepared_head=None,
                upstream_tree=None,
                last_error=str(error),
            )
            shutil.rmtree(transaction_path.parent, ignore_errors=True)
            raise ValueError(
                f"GitHub could not be reached for {source.display_name}. The "
                "subscribed deck was not changed and nothing was sent. "
                "Retrying is safe: "
                f"ankiops collab {kind} {source.display_name}"
            ) from error
    except _ConflictResolutionRequired:
        shutil.rmtree(transaction_path.parent, ignore_errors=True)
        raise
    except ValueError as error:
        state.save_collab_operation(
            source.source_path,
            operation_id,
            kind,
            "integrating",
            expected_head=original_commit,
            expected_fingerprint=original_fingerprint,
            prepared_head=None,
            upstream_tree=None,
            last_error=str(error),
        )
        shutil.rmtree(transaction_path.parent, ignore_errors=True)
        raise

    result_tree = transaction_git.tree("HEAD")
    if result_tree is None:
        raise ValueError(f"Could not prepare the update for {source.display_name}.")
    validate_collab_checkout(transaction_git, display_name=source.display_name)
    validate_collab_worktree(transaction_git, display_name=source.display_name)
    if result_tree == local_tree:
        transaction_commit = local_journal_commit
    else:
        transaction_commit = transaction_git.create_commit(
            result_tree,
            local_journal_commit,
            f"Integrate upstream changes for {source.display_name}",
        )
    state.save_collab_operation(
        source.source_path,
        operation_id,
        kind,
        "applying",
        expected_head=original_commit,
        expected_fingerprint=original_fingerprint,
        prepared_head=transaction_commit,
        upstream_tree=upstream_tree,
        recovery_ref=str(conflict_root) if conflict_root else None,
        last_error=None,
    )
    upstream_commit = transaction_git.run(
        ["rev-parse", f"upstream/{default_branch}"]
    ).stdout.strip()
    source_git.run(["fetch", str(transaction_git.root), transaction_commit])
    current_commit = source_git.head()
    current_fingerprint = _repository_fingerprint(source_git)
    if current_commit != original_commit or current_fingerprint != original_fingerprint:
        state.save_collab_operation(
            source.source_path,
            operation_id,
            kind,
            "integrating",
            expected_head=current_commit,
            expected_fingerprint=current_fingerprint,
            prepared_head=None,
            upstream_tree=None,
            recovery_ref=None,
            last_error="The subscribed deck changed while its update was prepared.",
        )
        shutil.rmtree(transaction_path.parent, ignore_errors=True)
        raise ValueError(
            f"The subscribed deck {source.display_name} changed while its update "
            "was prepared. The newer edit was left untouched. Retrying is safe: "
            f"ankiops collab {kind} {source.display_name}"
        )
    source_git.reset_hard("FETCH_HEAD")
    source_git.run(
        [
            "fetch",
            str(transaction_git.root),
            f"{upstream_commit}:{INTEGRATED_REF}",
            f"{upstream_commit}:refs/remotes/upstream/{default_branch}",
        ]
    )
    shutil.rmtree(transaction_path.parent, ignore_errors=True)
    if conflict_root:
        shutil.rmtree(conflict_root, ignore_errors=True)
    if upstream_tree is None:
        raise ValueError(
            f"Could not read the available update for {source.display_name}."
        )
    return operation_id, files_changed, saved_commit, upstream_tree, default_branch


def run_subscribe(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    collection_git = _require_collection_git(collection_root)
    source = _collab_source(collection_root, args.repository)
    if source.root.exists():
        raise ValueError(f"Collab source already exists: {source.display_name}")
    try:
        source_git = GitRepository.clone(
            str(source.github_url), source.root, anonymous=True
        )
        _copy_git_identity(collection_git, source_git)
        default_branch = source_git.default_branch("upstream")
        upstream_ref = f"upstream/{default_branch}"
        source_git.checkout_or_create_branch(JOURNAL_BRANCH, upstream_ref)
        source_git.run(["branch", "--unset-upstream"], check=False)
        source_git.update_ref(INTEGRATED_REF, upstream_ref)
        validate_collab_checkout(source_git, display_name=source.display_name)
        validate_collab_worktree(source_git, display_name=source.display_name)
        _ensure_submittable_note_keys(source)
    except BaseException:
        if source.root.exists():
            shutil.rmtree(source.root)
        raise
    _log_result(
        "subscribe",
        source,
        "downloaded",
        details=[f"Local repository: {source.root}"],
        next_steps=["Apply to Anki: ankiops fa"],
    )


def run_publish(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    source = _collab_source(collection_root, args.repository)
    state = SyncState.open(collection_root)
    operation_id, existing = _source_operation(state, source, "publish")
    retry_command = shlex.join(
        ["ankiops", "collab", "publish", args.deck, source.display_name]
    )
    state.save_collab_operation(
        source.source_path,
        operation_id,
        "publish",
        "publishing",
        recovery_ref=retry_command,
        last_error=None,
    )

    def record_prepared(prepared_head: str, root_fingerprint: str) -> None:
        state.save_collab_operation(
            source.source_path,
            operation_id,
            "publish",
            "publishing",
            prepared_head=prepared_head,
            expected_fingerprint=root_fingerprint,
        )

    try:
        publish_collab_deck(
            collection_root,
            args.deck,
            source,
            retry=existing is not None,
            expected_prepared_head=(
                str(existing["prepared_head"])
                if existing and existing.get("prepared_head")
                else None
            ),
            expected_root_fingerprint=(
                str(existing["expected_fingerprint"])
                if existing and existing.get("expected_fingerprint")
                else None
            ),
            record_prepared=record_prepared,
        )
    except RepositoryCollisionError as error:
        state.clear_collab_operation(source.source_path)
        raise ValueError(str(error)) from error
    except Exception as error:
        state.save_collab_operation(
            source.source_path,
            operation_id,
            "publish",
            "failed",
            last_error=str(error),
        )
        raise ValueError(
            f"Publishing did not finish for {source.display_name}. Your local deck "
            f"and any completed GitHub work were preserved. Retrying is safe: "
            f"{retry_command}. "
            f"Details: {error}"
        ) from error
    else:
        state.clear_collab_operation(source.source_path)
    finally:
        state.close()
    github_url = str(source.github_url).removesuffix(".git")
    _log_result(
        "publish",
        source,
        f"published '{args.deck}'",
        details=[
            f"Local repository: {source.root}",
            f"GitHub repository: {github_url}",
        ],
        next_steps=["Apply to Anki: ankiops fa"],
    )


def _restore_operation(
    state: SyncState,
    source: DeckSource,
    operation: dict[str, str | None],
) -> None:
    state.save_collab_operation(
        source.source_path,
        str(operation["operation_id"]),
        str(operation["kind"]),
        str(operation["state"]),
        expected_head=operation.get("expected_head"),
        expected_fingerprint=operation.get("expected_fingerprint"),
        prepared_head=operation.get("prepared_head"),
        upstream_tree=operation.get("upstream_tree"),
        recovery_ref=operation.get("recovery_ref"),
        publish_branch=operation.get("publish_branch"),
        pushed_sha=operation.get("pushed_sha"),
        pr_url=operation.get("pr_url"),
        last_error=operation.get("last_error"),
        requested_title=operation.get("requested_title"),
    )


def _live_pull_request(source: DeckSource, url: str) -> PullRequestInfo | None:
    if not url.startswith("https://github.com/"):
        return None
    return GitHubHost(source.root).pull_request(url)


def _github_slug_from_remote(url: str) -> str | None:
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
        r"([^/]+)/([^/]+?)(?:\.git)?/?",
        url,
        flags=re.IGNORECASE,
    )
    return f"{match.group(1)}/{match.group(2)}" if match else None


def _pull_request_head_changes(
    source: DeckSource,
    source_git: GitRepository,
    operation: dict[str, str | None],
    pull_request: PullRequestInfo | None,
) -> list[str]:
    if pull_request is None:
        return []
    changes = []
    expected_branch = operation.get("publish_branch")
    head_branch = str(getattr(pull_request, "head_branch", ""))
    if expected_branch and head_branch and head_branch != expected_branch:
        changes.append("head branch")
    expected_sha = operation.get("pushed_sha")
    head_sha = str(getattr(pull_request, "head_sha", ""))
    if expected_sha and head_sha and head_sha != expected_sha:
        changes.append("head commit")
    publish_url = source_git.remote_url("publish")
    publish_slug = _github_slug_from_remote(publish_url or "")
    head_owner = str(getattr(pull_request, "head_owner", ""))
    head_repository = str(getattr(pull_request, "head_repository", ""))
    if head_repository and not publish_slug:
        changes.append("configured publish repository")
    elif publish_slug:
        publish_owner, publish_name = publish_slug.split("/", 1)
        if head_owner and publish_owner.casefold() != head_owner.casefold():
            changes.append("head repository owner")
        if head_repository:
            repository_changed = publish_slug.casefold() != head_repository.casefold()
        else:
            source_name = source.display_name.split("/", 1)[1]
            repository_changed = publish_name.casefold() != source_name.casefold()
        if repository_changed:
            changes.append("configured publish repository")
    return changes


def _delete_submission_branch(
    source: DeckSource,
    source_git: GitRepository,
    operation: dict[str, str | None],
    pull_request: PullRequestInfo | None = None,
) -> bool:
    branch = operation.get("publish_branch")
    expected_sha = operation.get("pushed_sha")
    if not branch or not source_git.remote_url("publish"):
        return False
    remote_sha = source_git.remote_branch_sha("publish", str(branch))
    if remote_sha is None:
        return True
    if not expected_sha:
        return False
    head_changes = _pull_request_head_changes(
        source, source_git, operation, pull_request
    )
    if (
        pull_request
        and pull_request.state == "MERGED"
        and "head commit" in head_changes
    ):
        merged_head = str(getattr(pull_request, "head_sha", ""))
        integrated_head = source_git.ref_sha(INTEGRATED_REF)
        merged_head_available = bool(
            merged_head
            and source_git.run(
                ["cat-file", "-e", f"{merged_head}^{{commit}}"], check=False
            ).returncode
            == 0
        )
        if (
            merged_head_available
            and integrated_head
            and source_git.is_ancestor(merged_head, integrated_head)
        ):
            head_changes = [
                change for change in head_changes if change != "head commit"
            ]
            expected_sha = merged_head
    if head_changes:
        return False
    if remote_sha != expected_sha:
        return False
    try:
        source_git.delete_remote_branch_with_lease(
            "publish", str(branch), str(expected_sha)
        )
    except subprocess.CalledProcessError as error:
        remote_sha = source_git.remote_branch_sha("publish", str(branch))
        if remote_sha is None:
            return True
        if remote_sha != expected_sha:
            return False
        raise ValueError(f"GitHub branch {branch} could not be removed.") from error
    if source_git.remote_branch_sha("publish", str(branch)) is not None:
        raise ValueError(f"GitHub branch {branch} could not be removed.")
    return True


def _update_one(collection_root: Path, source: DeckSource, state: SyncState) -> None:
    source_git = _require_source_git(source)
    anki_manifest_before = _read_anki_manifest(source)
    integrated_before = source_git.ref_sha(INTEGRATED_REF)
    pending_action = state.get_collab_operation(source.source_path)
    uploaded_tree_before = source_git.tree(UPLOADED_REF)
    operation_kind = str(pending_action["kind"]) if pending_action else "update"
    pull_request: PullRequestInfo | None = None
    pull_request_state: str | None = None
    pull_request_merged = False
    pull_request_unavailable = False
    if pending_action and pending_action.get("pr_url"):
        try:
            pull_request = _live_pull_request(source, str(pending_action["pr_url"]))
            pull_request_state = pull_request.state if pull_request else None
            pull_request_merged = pull_request_state == "MERGED"
        except ValueError as error:
            pull_request_unavailable = True
            logger.debug(
                "Could not read pull request for %s: %s", source.display_name, error
            )
    pull_request_head_changed = bool(
        pending_action
        and pending_action["kind"] == "submit"
        and pull_request_state == "OPEN"
        and _pull_request_head_changes(source, source_git, pending_action, pull_request)
    )
    branch_cleaned = False
    if pull_request_merged and pending_action and pending_action["kind"] == "submit":
        branch_cleaned = _delete_submission_branch(
            source, source_git, pending_action, pull_request
        )
    post_submission_states = {"ready", "push_failed", "pushed", "pr_failed", "pr_open"}
    resolving_open_submission = bool(
        pending_action
        and pending_action["kind"] == "submit"
        and pending_action["state"] == "conflict"
        and pending_action.get("pr_url")
    )
    try:
        _operation_id, _files_changed, saved_commit, upstream_tree, _default_branch = (
            _integrate_upstream(
                collection_root,
                source,
                source_git,
                state,
                kind=operation_kind,
                uploaded_accepted=pull_request_merged,
            )
        )
    except (ValueError, subprocess.CalledProcessError):
        current_action = state.get_collab_operation(source.source_path)
        if (
            pending_action
            and pending_action["state"] in post_submission_states
            and current_action
            and current_action["state"] != "conflict"
        ):
            _restore_operation(state, source, pending_action)
        raise
    upstream_changed = bool(
        integrated_before and source_git.tree(integrated_before) != upstream_tree
    )
    changed_paths: list[GitPathChange] = []
    anki_applicable_changes = False
    if upstream_changed and integrated_before:
        changed_paths = source_git.diff_name_status(integrated_before, INTEGRATED_REF)
        flat_changed_paths = {
            path for changed_path in changed_paths for path in changed_path.paths
        }
        anki_manifest_after = _read_anki_manifest(source)
        if anki_manifest_before is not None and anki_manifest_after is not None:
            anki_applicable_changes = anki_applicable_paths_changed(
                flat_changed_paths,
                anki_manifest_before,
                anki_manifest_after,
            )
        else:
            anki_applicable_changes = _conservative_anki_guidance(flat_changed_paths)
    local_contribution = source_git.tree("HEAD") != upstream_tree
    if (
        pending_action
        and pending_action["kind"] == "submit"
        and (
            pending_action["state"] in post_submission_states
            or resolving_open_submission
        )
    ):
        submission_accepted = pull_request_merged or bool(
            not pull_request_unavailable
            and pull_request_state is None
            and integrated_before
            and uploaded_tree_before
            and _tree_delta_is_present(
                source_git,
                base_tree=source_git.tree(integrated_before) or integrated_before,
                changed_tree=uploaded_tree_before,
                candidate_tree=upstream_tree,
            )
        )
        if submission_accepted:
            if not branch_cleaned:
                branch_cleaned = _delete_submission_branch(
                    source, source_git, pending_action, pull_request
                )
            state.clear_collab_operation(source.source_path)
            source_git.delete_ref(UPLOADED_REF)
        else:
            if resolving_open_submission:
                resumed_action = {
                    **pending_action,
                    "state": "pr_open",
                    "recovery_ref": None,
                    "last_error": None,
                }
                _restore_operation(state, source, resumed_action)
            else:
                _restore_operation(state, source, pending_action)
    else:
        state.clear_collab_operation(source.source_path)
    remaining_action = state.get_collab_operation(source.source_path)

    if upstream_changed and saved_commit:
        summary = "committed local changes and integrated upstream changes"
    elif upstream_changed:
        summary = "integrated upstream changes"
    elif saved_commit:
        summary = "committed local changes; upstream already current"
    elif local_contribution:
        summary = "no upstream changes"
    else:
        summary = "already up to date"

    details = []
    if upstream_changed:
        details.append(f"Files changed: {_counted(len(changed_paths), 'path')}")
    if local_contribution:
        details.append("Local contribution: ready to submit")
    if pull_request_head_changed:
        details.append("Pull request: changed on GitHub since your last upload")
    if pull_request_merged and pending_action and not branch_cleaned:
        details.append("Submission branch: kept because it changed on GitHub")

    next_steps = []
    if upstream_changed and anki_applicable_changes:
        next_steps.append("Apply to Anki: ankiops fa")
    if remaining_action and remaining_action.get("pr_url"):
        if pull_request_head_changed:
            next_steps.append(
                "Inspect GitHub changes before another upload: "
                f"{remaining_action['pr_url']}"
            )
        elif local_contribution:
            action = (
                "Submit again"
                if pull_request_state == "CLOSED"
                else "Update pull request"
            )
            next_steps.append(f"{action}: ankiops collab submit {source.display_name}")
    else:
        if remaining_action and remaining_action["kind"] == "submit":
            next_steps.append(
                f"Retry submission: ankiops collab submit {source.display_name}"
            )
        elif local_contribution:
            next_steps.append(
                f"Submit contribution: ankiops collab submit {source.display_name}"
            )

    _log_result(
        "update",
        source,
        summary,
        details=details,
        next_steps=next_steps,
    )


def run_update(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    sources = discover_deck_sources(collection_root)[1:]
    if getattr(args, "repository", None):
        requested_source = _collab_source(collection_root, args.repository)
        if not requested_source.root.exists():
            raise ValueError(f"Unknown collab source: {requested_source.display_name}")
        sources = [requested_source]
    if not sources:
        output_line("No subscribed collab decks.")
        _log_next_steps(["Subscribe to a deck: ankiops collab subscribe OWNER/REPO"])
        return
    state = SyncState.open(collection_root)
    failures = []
    try:
        for index, source in enumerate(sources):
            if index:
                output_line()
            try:
                _update_one(collection_root, source, state)
            except (ValueError, subprocess.CalledProcessError) as error:
                failures.append(f"{source.display_name}: {error}")
    finally:
        state.close()
    if failures:
        raise ValueError(
            f"Collab update finished with {len(failures)} failure(s): "
            + " | ".join(failures)
        )


def run_submit(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    source = _collab_source(collection_root, args.repository)
    source_git = _require_source_git(source)
    _ensure_submittable_note_keys(source)
    requested_title = getattr(args, "title", None)
    state = SyncState.open(collection_root)
    try:
        pending_action = state.get_collab_operation(source.source_path)
        if (
            requested_title is None
            and pending_action
            and pending_action["kind"] == "submit"
            and pending_action.get("requested_title")
        ):
            requested_title = str(pending_action["requested_title"])
        if pending_action and pending_action["kind"] == "update":
            raise ValueError(
                f"An update for {source.display_name} still needs attention. Run: "
                f"ankiops collab update {source.display_name}"
            )
        if pending_action and pending_action["state"] == "conflict":
            raise ValueError(
                f"A contribution for {source.display_name} has unresolved upstream "
                f"changes. Run: ankiops collab update {source.display_name}"
            )
        pending_pr_url = (
            str(pending_action["pr_url"])
            if pending_action
            and pending_action["kind"] == "submit"
            and pending_action.get("pr_url")
            else None
        )
        pull_request: PullRequestInfo | None = None
        if pending_pr_url:
            try:
                pull_request = _live_pull_request(source, pending_pr_url)
                pull_request_state = pull_request.state if pull_request else None
            except ValueError as error:
                raise ValueError(
                    f"Could not confirm the pull request for {source.display_name}. "
                    "Nothing was sent; retrying is safe. "
                    f"Details: {error}"
                ) from error
            if pull_request_state == "MERGED":
                raise ValueError(
                    f"The pull request for {source.display_name} has been merged. "
                    f"Run: ankiops collab update {source.display_name}"
                )
            if pull_request_state == "CLOSED":
                _delete_submission_branch(
                    source, source_git, pending_action, pull_request
                )
                state.clear_collab_operation(source.source_path)
                source_git.delete_ref(UPLOADED_REF)
                pending_action = None
                pending_pr_url = None
                pull_request = None
        draft_changed = bool(
            pending_action
            and (
                source_git.status_lines()
                or source_git.head() != pending_action.get("prepared_head")
            )
        )
        pull_request_head_changes = (
            _pull_request_head_changes(source, source_git, pending_action, pull_request)
            if pending_pr_url and pending_action
            else []
        )
        if pull_request_head_changes:
            if draft_changed or requested_title:
                raise ValueError(
                    f"The open pull request for {source.display_name} changed on "
                    "GitHub since AnkiOps last uploaded it. Nothing was sent, so "
                    "the GitHub changes were preserved. Inspect the pull request "
                    "and let its maintainer merge or close it before uploading "
                    f"another revision: {pending_pr_url}"
                )
            _log_result(
                "submit",
                source,
                "pull request changed on GitHub",
                details=[
                    "Changed: " + ", ".join(pull_request_head_changes),
                    f"Pull request: {pending_pr_url}",
                ],
            )
            return
        pull_request_owner = str(getattr(pull_request, "head_owner", "")) or None
        if pending_pr_url and pull_request_owner:
            github = GitHubHost(source.root)
            active_login = github.login()
            if active_login.casefold() != pull_request_owner.casefold():
                publish_url = source_git.remote_url("publish") or ""
                publish_slug = _github_slug_from_remote(publish_url) or (
                    f"{pull_request_owner}/{source.display_name.split('/', 1)[1]}"
                )
                try:
                    can_push = github.can_push(publish_slug)
                except ValueError as error:
                    raise ValueError(
                        f"Could not confirm GitHub push access to {publish_slug}. "
                        "Nothing was sent. Retrying is safe: ankiops collab submit "
                        f"{source.display_name}. Details: {error}"
                    ) from error
                if not can_push:
                    raise ValueError(
                        f"GitHub CLI is authenticated as @{active_login}, but this "
                        f"pull request belongs to @{pull_request_owner}. Nothing was "
                        "sent. Switch accounts, then retry: gh auth switch --hostname "
                        f"github.com --user {pull_request_owner}"
                    )
        if (
            pending_pr_url
            and pending_action
            and pending_action["state"] == "pr_open"
            and not draft_changed
        ):
            if requested_title:
                state.save_collab_operation(
                    source.source_path,
                    str(pending_action["operation_id"]),
                    "submit",
                    "pr_open",
                    requested_title=requested_title,
                )
                try:
                    GitHubHost(source.root).update_pr(
                        pending_pr_url, title=requested_title
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Could not update the pull request title for "
                        f"{source.display_name}. The pull request remains open; "
                        f"retrying is safe. Details: {error}"
                    ) from error
                state.save_collab_operation(
                    source.source_path,
                    str(pending_action["operation_id"]),
                    "submit",
                    "pr_open",
                    requested_title=None,
                    last_error=None,
                )
                _log_result(
                    "submit",
                    source,
                    "pull request updated",
                    details=[
                        f"Title: {requested_title}",
                        f"Pull request: {pending_pr_url}",
                    ],
                )
            else:
                _log_result(
                    "submit",
                    source,
                    "pull request already open",
                    details=[f"Pull request: {pending_pr_url}"],
                )
            return

        saved_commit: str | None = None
        if (
            pending_action is None
            or pending_action["state"] in {"integrating", "applying"}
            or (pending_pr_url and draft_changed)
        ):
            (
                operation_id,
                _files_changed,
                saved_commit,
                upstream_tree,
                target_branch,
            ) = _integrate_upstream(
                collection_root,
                source,
                source_git,
                state,
                kind="submit",
                requested_title=requested_title,
            )
            if source_git.tree("HEAD") == upstream_tree:
                _log_result(
                    "submit",
                    source,
                    "nothing to submit",
                    details=(
                        ["Local changes: committed; content matches upstream"]
                        if saved_commit
                        else None
                    ),
                )
                state.clear_collab_operation(source.source_path)
                return
            contribution_branch = f"ankiops/{operation_id}"
            local_commit = source_git.head()
            if local_commit is None:
                raise ValueError(f"Collab source {source.display_name} has no commits.")
            state.save_collab_operation(
                source.source_path,
                operation_id,
                "submit",
                "ready",
                prepared_head=local_commit,
                upstream_tree=upstream_tree,
                publish_branch=contribution_branch,
                pushed_sha=(
                    pending_action.get("pushed_sha")
                    if pending_pr_url and pending_action
                    else None
                ),
                pr_url=pending_pr_url,
                last_error=None,
                requested_title=requested_title,
            )
        else:
            if source_git.status_lines() or source_git.head() != pending_action.get(
                "prepared_head"
            ):
                raise ValueError(
                    f"A contribution for {source.display_name} is waiting to finish, "
                    "but the collab files changed afterward. Nothing was sent. "
                    f"Run: ankiops collab status {source.display_name}"
                )
            operation_id = str(pending_action["operation_id"])
            contribution_branch = str(pending_action["publish_branch"])
            upstream_tree = str(pending_action["upstream_tree"])
            target_branch = source_git.default_branch("upstream")

        local_commit = source_git.head()
        if local_commit is None:
            raise ValueError(f"Collab source {source.display_name} has no commits.")
        local_tree = source_git.tree(local_commit)
        if local_tree is None:
            raise ValueError(
                f"Could not read the local contribution for {source.display_name}."
            )
        upstream_parent = source_git.ref_sha(INTEGRATED_REF)
        if upstream_parent is None or source_git.tree(upstream_parent) != upstream_tree:
            raise ValueError(
                f"Could not resolve the integrated upstream checkpoint for "
                f"{source.display_name}. Nothing was sent. Run: "
                f"ankiops collab update {source.display_name}"
            )
        title = requested_title or _derive_submit_title(source_git, upstream_parent)
        snapshot_commit = source_git.create_commit(
            local_tree,
            upstream_parent,
            title,
        )
        github = GitHubHost(source.root)
        try:
            if pending_pr_url:
                publish_url = source_git.remote_url("publish")
                if publish_url is None:
                    raise ValueError(
                        "The existing pull request has no publish repository."
                    )
                contribution_slug = _github_slug_from_remote(publish_url) or (
                    f"{pull_request_owner}/{source.display_name.split('/', 1)[1]}"
                    if pull_request_owner
                    else source.display_name
                )
                contributor = (
                    pull_request_owner
                    if pull_request_owner
                    else contribution_slug.split("/", 1)[0]
                )
            else:
                contribution_slug, contributor = github.publish_target(
                    source.display_name
                )
        except ValueError as error:
            state.save_collab_operation(
                source.source_path,
                operation_id,
                "submit",
                "ready",
                last_error=str(error),
            )
            raise ValueError(
                f"Your contribution for {source.display_name} was committed locally "
                "but GitHub setup did not finish. Nothing was uploaded and private "
                "decks were not touched. Retrying is safe: "
                f"ankiops collab submit {source.display_name}. Details: {error}"
            ) from error
        if not pending_pr_url:
            source_git.set_remote(
                "publish", f"https://github.com/{contribution_slug}.git"
            )
        uploaded_commit = source_git.remote_branch_sha("publish", contribution_branch)
        if (
            uploaded_commit
            and pending_action
            and source_git.tree(uploaded_commit) == local_tree
        ):
            # A previous attempt may have completed the push before interruption.
            snapshot_commit = uploaded_commit
        if uploaded_commit != snapshot_commit:
            try:
                expected_sha = (
                    str(pending_action["pushed_sha"])
                    if pending_pr_url
                    and pending_action
                    and pending_action.get("pushed_sha")
                    else None
                )
                if expected_sha:
                    source_git.push_force_with_lease(
                        "publish",
                        snapshot_commit,
                        contribution_branch,
                        expected_sha,
                    )
                else:
                    source_git.push(
                        "publish",
                        snapshot_commit,
                        contribution_branch,
                    )
            except subprocess.CalledProcessError as error:
                state.save_collab_operation(
                    source.source_path,
                    operation_id,
                    "submit",
                    "push_failed",
                    prepared_head=local_commit,
                    upstream_tree=upstream_tree,
                    publish_branch=contribution_branch,
                    last_error=str(error),
                )
                raise ValueError(
                    f"Your contribution for {source.display_name} was committed "
                    "locally, but it did not reach GitHub. Private decks were not "
                    "touched. Retrying is safe: ankiops collab submit "
                    f"{source.display_name}"
                ) from error

        state.save_collab_operation(
            source.source_path,
            operation_id,
            "submit",
            "pushed",
            prepared_head=local_commit,
            upstream_tree=upstream_tree,
            publish_branch=contribution_branch,
            pushed_sha=snapshot_commit,
            last_error=None,
        )
        source_git.update_ref(UPLOADED_REF, snapshot_commit)
        contribution_head = f"{contributor}:{contribution_branch}"
        try:
            if pending_pr_url:
                if requested_title and pending_pr_url.startswith("https://github.com/"):
                    github.update_pr(pending_pr_url, title=title)
                pr_url = pending_pr_url
            else:
                pr_url = github.create_pr(
                    source.display_name,
                    head=contribution_head,
                    base=target_branch,
                    title=title,
                    body=f"Submitted with AnkiOps from {source.display_name}.",
                )
        except ValueError as error:
            state.save_collab_operation(
                source.source_path,
                operation_id,
                "submit",
                "pr_failed",
                prepared_head=local_commit,
                upstream_tree=upstream_tree,
                publish_branch=contribution_branch,
                pushed_sha=snapshot_commit,
                last_error=str(error),
            )
            raise ValueError(
                f"Your contribution reached GitHub, but its pull request was not "
                "created. No private files changed. Retrying is safe: "
                "ankiops collab submit "
                f"{source.display_name}. Details: {error}"
            ) from error
        state.save_collab_operation(
            source.source_path,
            operation_id,
            "submit",
            "pr_open",
            prepared_head=local_commit,
            upstream_tree=upstream_tree,
            publish_branch=contribution_branch,
            pushed_sha=snapshot_commit,
            pr_url=pr_url,
            last_error=None,
            requested_title=None,
        )
        _log_result(
            "submit",
            source,
            "pull request updated" if pending_pr_url else "pull request opened",
            details=[
                f"Commit: {title}",
                f"GitHub repository: https://github.com/{contribution_slug}",
                f"Pull request: {pr_url}",
            ],
        )
    finally:
        state.close()


def _repository_state(source_git: GitRepository) -> _RepositoryState:
    _default_branch, upstream_ref = _upstream_ref(source_git)
    upstream_tree = source_git.tree(upstream_ref)
    if upstream_tree is None:
        raise ValueError("Could not read the upstream content tree.")
    if source_git.worktree_matches(upstream_ref):
        return _RepositoryState(_RepositoryRelation.CURRENT, 0, 0)
    integrated_ref = source_git.ref_sha(INTEGRATED_REF)
    if integrated_ref is None:
        raise ValueError("The collab source has no integrated upstream checkpoint.")
    integrated_tree = source_git.tree(integrated_ref)
    if integrated_tree is None:
        raise ValueError("Could not read the integrated content tree.")
    local_content = not source_git.worktree_matches(INTEGRATED_REF)
    if source_git.is_ancestor(integrated_ref, upstream_ref):
        upstream_commits = int(
            source_git.run(
                ["rev-list", "--count", f"{integrated_ref}..{upstream_ref}"]
            ).stdout.strip()
        )
    else:
        upstream_commits = int(
            source_git.run(
                ["rev-list", "--count", upstream_ref, f"^{integrated_ref}"]
            ).stdout.strip()
        )
    local_commits = 1 if local_content else 0
    if local_content and upstream_commits:
        relation = _RepositoryRelation.DIVERGED
    elif local_content:
        relation = _RepositoryRelation.AHEAD
    elif upstream_commits:
        relation = _RepositoryRelation.BEHIND
    else:
        relation = _RepositoryRelation.CURRENT
    return _RepositoryState(relation, local_commits, upstream_commits)


def _status_one(
    source: DeckSource,
    state: SyncState,
) -> None:
    source_git = _require_source_git(source)
    local_changes = source_git.status_lines()
    repository_state: _RepositoryState | None = None
    try:
        source_git.fetch("upstream")
        repository_state = _repository_state(source_git)
    except (ValueError, subprocess.CalledProcessError) as error:
        logger.debug("Could not check GitHub for %s: %s", source.display_name, error)
    pending_action = state.get_collab_operation(source.source_path)
    pull_request: PullRequestInfo | None = None
    pull_request_state: str | None = None
    pull_request_unavailable = False
    if pending_action and pending_action.get("pr_url"):
        try:
            pull_request = _live_pull_request(source, str(pending_action["pr_url"]))
            pull_request_state = pull_request.state if pull_request else None
        except ValueError as error:
            pull_request_unavailable = True
            logger.debug(
                "Could not read pull request for %s: %s", source.display_name, error
            )
    pull_request_head_changed = bool(
        pending_action
        and pending_action["kind"] == "submit"
        and _pull_request_head_changes(source, source_git, pending_action, pull_request)
    )
    open_submission_changed = bool(
        pending_action
        and pending_action["kind"] == "submit"
        and pending_action["state"] == "pr_open"
        and pending_action.get("pr_url")
        and (local_changes or source_git.head() != pending_action.get("prepared_head"))
    )
    open_submission_current = bool(
        pending_action
        and pending_action["kind"] == "submit"
        and pending_action["state"] == "pr_open"
        and not open_submission_changed
        and not pull_request_head_changed
    )
    submission_accepted = bool(
        not pull_request_unavailable
        and not local_changes
        and repository_state
        and repository_state.relation is _RepositoryRelation.CURRENT
        and pending_action
        and pending_action["kind"] == "submit"
        and pull_request_state is None
    )

    output_line(source.display_name)
    working_tree = (
        "clean" if not local_changes else _counted(len(local_changes), "changed path")
    )
    output_line(f"  Working tree: {working_tree}")
    for line in local_changes:
        output_line(f"    {line}")

    if repository_state is None:
        output_line("  Local contribution: unknown")
        output_line("  Upstream: unavailable")
    elif repository_state.relation is _RepositoryRelation.CURRENT:
        output_line("  Local contribution: none")
        output_line("  Upstream: up to date")
    elif repository_state.relation is _RepositoryRelation.AHEAD:
        if pull_request_head_changed:
            contribution_state = "preserved locally"
        else:
            contribution_state = (
                "uploaded" if open_submission_current else "ready to submit"
            )
        output_line(
            "  Local contribution: "
            f"{_counted(repository_state.local_commits, 'change')} "
            f"{contribution_state}"
        )
        output_line("  Upstream: up to date")
    elif repository_state.relation is _RepositoryRelation.BEHIND:
        output_line("  Local contribution: none")
        output_line(
            "  Upstream: "
            f"{_counted(repository_state.upstream_commits, 'update')} available"
        )
    else:
        if pull_request_head_changed:
            contribution_state = "preserved locally"
        else:
            contribution_state = (
                "uploaded" if open_submission_current else "ready to submit"
            )
        output_line(
            "  Local contribution: "
            f"{_counted(repository_state.local_commits, 'change')} "
            f"{contribution_state}"
        )
        output_line(
            f"  Upstream: {_counted(repository_state.upstream_commits, 'update')} "
            "available; update before submitting"
        )
    if pending_action:
        if pending_action["state"] == "conflict":
            output_line(
                "  Update: conflict requires resolution at "
                f"{pending_action.get('recovery_ref')}"
            )
        elif pending_action["kind"] == "submit":
            if pull_request_state == "MERGED" or submission_accepted:
                submission_state = "merged; update to integrate and clean up"
            elif pull_request_state == "CLOSED":
                submission_state = "pull request closed without merge"
            elif pull_request_unavailable:
                submission_state = "pull request state unavailable"
            elif pull_request_head_changed:
                submission_state = (
                    "pull request open; changed on GitHub since your last upload"
                )
            elif open_submission_changed:
                submission_state = "pull request open; local changes not uploaded"
            else:
                submission_state = {
                    "ready": "ready to upload",
                    "push_failed": "not uploaded",
                    "pushed": "uploaded; pull request not confirmed",
                    "pr_failed": "uploaded; pull request not created",
                    "pr_open": "pull request open",
                }.get(str(pending_action["state"]), "interrupted")
            output_line(f"  Submission: {submission_state}")
        else:
            output_line(
                f"  Interrupted action: {pending_action['kind']} "
                f"{pending_action['state']}"
            )
        if pending_action.get("pr_url"):
            output_line(f"  Pull request: {pending_action['pr_url']}")

    next_steps = []
    if repository_state is None:
        next_steps.append(f"Retry status: ankiops collab status {source.display_name}")
    elif pending_action and (
        pending_action["state"] == "conflict" or pending_action["kind"] == "update"
    ):
        next_steps.append(f"Resume update: ankiops collab update {source.display_name}")
    elif repository_state.relation in {
        _RepositoryRelation.BEHIND,
        _RepositoryRelation.DIVERGED,
    }:
        next_steps.append(
            f"Integrate upstream: ankiops collab update {source.display_name}"
        )
    elif pull_request_unavailable:
        next_steps.append(f"Retry status: ankiops collab status {source.display_name}")
    elif pull_request_state == "CLOSED":
        next_steps.append(f"Submit again: ankiops collab submit {source.display_name}")
    elif pull_request_state == "MERGED" or submission_accepted:
        next_steps.append(
            f"Integrate merge: ankiops collab update {source.display_name}"
        )
    elif pull_request_head_changed and pending_action and pending_action.get("pr_url"):
        next_steps.append(
            f"Inspect GitHub changes before another upload: {pending_action['pr_url']}"
        )
    elif open_submission_changed:
        next_steps.append(
            f"Update pull request: ankiops collab submit {source.display_name}"
        )
    elif not (pending_action and pending_action.get("pr_url")):
        if (
            pending_action
            and pending_action["kind"] == "submit"
            and not pending_action.get("pr_url")
        ):
            next_steps.append(
                f"Retry submission: ankiops collab submit {source.display_name}"
            )
        elif local_changes or repository_state.relation is _RepositoryRelation.AHEAD:
            next_steps.append(
                f"Submit contribution: ankiops collab submit {source.display_name}"
            )
    _log_next_steps(next_steps)


def _status_publish_operation(
    source: DeckSource,
    operation: dict[str, str | None],
) -> None:
    output_line(source.display_name)
    output_line(f"  Publish: {operation['state']}")
    local_repository = str(source.root) if source.root.exists() else "not prepared"
    output_line(f"  Local repository: {local_repository}")
    if operation.get("last_error"):
        output_line(f"  Last error: {operation['last_error']}")
    retry_command = operation.get("recovery_ref")
    if retry_command:
        _log_next_steps([f"Retry publish: {retry_command}"])


def run_status(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    sources = discover_deck_sources(collection_root)[1:]
    requested_source: DeckSource | None = None
    if getattr(args, "repository", None):
        requested_source = _collab_source(collection_root, args.repository)
        sources = [requested_source] if requested_source.root.exists() else []
    state = SyncState.open(collection_root)
    try:
        publish_operations = {
            str(operation["source_path"]): operation
            for operation in state.list_collab_operations()
            if operation["kind"] == "publish"
        }
        if requested_source is not None and not requested_source.root.exists():
            operation = publish_operations.get(requested_source.source_path)
            if not operation:
                raise ValueError(
                    f"Unknown collab source: {requested_source.display_name}"
                )
            _status_publish_operation(requested_source, operation)
            return
        if requested_source is None:
            sources_by_path = {source.source_path: source for source in sources}
            for source_path in publish_operations:
                if source_path not in sources_by_path:
                    repository = source_path.removeprefix("collab/")
                    sources_by_path[source_path] = DeckSource.collab(
                        collection_root, repository
                    )
            sources = [sources_by_path[path] for path in sorted(sources_by_path)]
        if not sources:
            output_line("No subscribed collab decks.")
            _log_next_steps(
                ["Subscribe to a deck: ankiops collab subscribe OWNER/REPO"]
            )
            return
        for index, source in enumerate(sources):
            if index:
                output_line()
            operation = publish_operations.get(source.source_path)
            if operation:
                _status_publish_operation(source, operation)
            else:
                _status_one(source, state)
    finally:
        state.close()


def run(args: SimpleNamespace) -> None:
    handlers = {
        "publish": run_publish,
        "subscribe": run_subscribe,
        "update": run_update,
        "submit": run_submit,
        "status": run_status,
    }
    handler = handlers.get(args.collab_command)
    if handler is None:
        raise SystemExit(2)
    handler(args)

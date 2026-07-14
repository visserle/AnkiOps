"""Independent-repository collab source commands."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

from ankiops.collab.errors import RepositoryCollisionError
from ankiops.collab.git_state import (
    CONFLICT_BASE_REF,
    CONFLICT_LOCAL_REF,
    CONFLICT_UPSTREAM_REF,
    CONTRIBUTION_BRANCH,
    INTEGRATED_REF,
    JOURNAL_BRANCH,
    SUBMISSION_REF,
)
from ankiops.collab.hosting import GitHubHost, PullRequestInfo
from ankiops.collab.publish import prepared_publish, publish_collab_deck
from ankiops.collab.source_security import (
    validate_collab_checkout,
)
from ankiops.collection import require_collection_root
from ankiops.console import print_line as output_line
from ankiops.console import print_next_steps as output_next_steps
from ankiops.console import print_result as output_result
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    is_deck_markdown_filename,
    parse_github_slug,
)
from ankiops.git import GitPathChange, GitRepository
from ankiops.interchange import ParsedSource, parse_source, require_note_keys

logger = logging.getLogger(__name__)

_UNRESOLVED_CONFLICT_PLACEHOLDER = (
    b"<<<<<<< AnkiOps unresolved conflict\n"
    b"Git could not create a textual merge for this path. Raw versions are stored "
    b"beside\nthis file as .base, .local, and .upstream when that version exists.\n"
    b"=======\n"
    b"Replace this entire placeholder with the desired final contents, or delete "
    b"this file\nto resolve the conflict by deleting the path, before retrying.\n"
    b">>>>>>> AnkiOps unresolved conflict\n"
)


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


@dataclass(frozen=True)
class _ConflictState:
    kind: str
    requested_title: str | None
    root: Path
    base_commit: str
    local_commit: str
    upstream_commit: str


@dataclass(frozen=True)
class _IntegrationResult:
    saved_commit: str | None
    parsed_source: ParsedSource
    newer_upstream: bool = False
    resolved_conflict: bool = False


class _SubmissionPhase(Enum):
    NONE = auto()
    PREPARED = auto()
    PR_OPEN = auto()
    MERGED = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class _SubmissionState:
    phase: _SubmissionPhase
    snapshot: str | None
    publish_slug: str | None
    pull_request: PullRequestInfo | None


def _collab_source(collection_root: Path, repository: str) -> DeckSource:
    return DeckSource.collab(collection_root, repository)


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


def _derive_submit_title(source_git: GitRepository, upstream_ref: str) -> str:
    changes = source_git.diff_name_status(upstream_ref, "HEAD")
    deck_changes: list[tuple[str, tuple[Path, ...]]] = []
    assets_changed = False
    for change in changes:
        deck_paths = []
        for raw_path in change.paths:
            path = Path(raw_path)
            if len(path.parts) == 1 and is_deck_markdown_filename(path.name):
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


def _conflict_directory(collection_root: Path, source: DeckSource) -> Path:
    owner, repo_name = source.display_name.split("/", 1)
    return collection_root / ".ankiops" / "conflicts" / owner / repo_name


def _conflict_retry_command(source: DeckSource, kind: str) -> str:
    return shlex.join(["ankiops", "collab", kind, source.display_name])


def _load_conflict_state(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
) -> _ConflictState | None:
    root = _conflict_directory(collection_root, source)
    metadata_path = root / "conflict.json"
    commits = {
        "base": source_git.ref_sha(CONFLICT_BASE_REF),
        "local": source_git.ref_sha(CONFLICT_LOCAL_REF),
        "upstream": source_git.ref_sha(CONFLICT_UPSTREAM_REF),
    }
    if not metadata_path.exists() and not any(commits.values()):
        return None
    if metadata_path.exists() and not any(commits.values()):
        shutil.rmtree(root)
        return None
    if not metadata_path.exists() or not all(commits.values()):
        raise ValueError(
            f"The saved conflict for {source.display_name} is incomplete. "
            f"Inspect {root}; the subscribed repository was left untouched."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        kind = metadata["kind"]
        requested_title = metadata.get("requested_title")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"The saved conflict metadata for {source.display_name} is invalid: "
            f"{metadata_path}"
        ) from error
    if kind not in {"update", "submit"} or (
        requested_title is not None and not isinstance(requested_title, str)
    ):
        raise ValueError(
            f"The saved conflict metadata for {source.display_name} is invalid: "
            f"{metadata_path}"
        )
    return _ConflictState(
        kind=kind,
        requested_title=requested_title,
        root=root,
        base_commit=str(commits["base"]),
        local_commit=str(commits["local"]),
        upstream_commit=str(commits["upstream"]),
    )


def _write_conflict_metadata(
    conflict_root: Path,
    *,
    kind: str,
    requested_title: str | None,
) -> None:
    (conflict_root / "conflict.json").write_text(
        json.dumps(
            {"kind": kind, "requested_title": requested_title},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _clear_conflict_state(
    source_git: GitRepository,
    conflict_root: Path,
) -> None:
    source_git.delete_refs(
        (CONFLICT_BASE_REF, CONFLICT_LOCAL_REF, CONFLICT_UPSTREAM_REF)
    )
    shutil.rmtree(conflict_root, ignore_errors=True)


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
    *,
    kind: str,
    requested_title: str | None,
) -> Path:
    conflict_root = _conflict_directory(collection_root, source)
    shutil.rmtree(conflict_root, ignore_errors=True)
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
    _write_conflict_metadata(
        conflict_root,
        kind=kind,
        requested_title=requested_title,
    )
    return conflict_root


def _upstream_ref(source_git: GitRepository) -> str:
    return f"upstream/{source_git.default_branch('upstream')}"


def _contains_conflict_markers(content: bytes) -> bool:
    return all(marker in content for marker in (b"<<<<<<<", b"=======", b">>>>>>>"))


def _apply_preserved_resolutions(
    repository: GitRepository, conflict_root: Path
) -> list[str]:
    unresolved = []
    for rel_path in repository.unmerged_paths():
        resolution = conflict_root / rel_path
        if not resolution.exists():
            repository.run(["rm", "-f", "--", rel_path])
            continue
        resolution_content = resolution.read_bytes()
        if _contains_conflict_markers(resolution_content):
            unresolved.append(rel_path)
            continue
        target = repository.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resolution_content)
        repository.run(["add", "--", rel_path])
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


def _finish_content_merge(
    source: DeckSource,
    source_git: GitRepository,
    *,
    local_commit: str,
    result_tree: str,
    upstream_commit: str,
) -> ParsedSource:
    local_tree = source_git.tree(local_commit)
    if local_tree is None:
        raise ValueError(
            f"Could not read the local checkpoint for {source.display_name}."
        )
    if result_tree == local_tree:
        source_git.reset_hard(local_commit)
    else:
        result_commit = source_git.create_commit(
            result_tree,
            local_commit,
            f"Integrate upstream changes for {source.display_name}",
        )
        source_git.reset_hard(result_commit)
    parsed_source = parse_source(source)
    source_git.update_ref(INTEGRATED_REF, upstream_commit)
    return parsed_source


def _resolve_frozen_conflict(
    source: DeckSource,
    source_git: GitRepository,
    conflict: _ConflictState,
) -> _IntegrationResult:
    if source_git.head() != conflict.local_commit or source_git.status_lines():
        raise ValueError(
            f"A conflict is waiting for resolution at {conflict.root}. Do not edit "
            f"the subscribed source while it is pending. Restore it to the committed "
            f"local checkpoint, then retry: "
            f"{_conflict_retry_command(source, conflict.kind)}"
        )
    base_tree = source_git.tree(conflict.base_commit)
    local_tree = source_git.tree(conflict.local_commit)
    upstream_tree = source_git.tree(conflict.upstream_commit)
    if not base_tree or not local_tree or not upstream_tree:
        raise ValueError(
            f"The saved conflict commits for {source.display_name} are incomplete. "
            f"Inspect {conflict.root}; the subscribed repository was left untouched."
        )
    try:
        try:
            result_tree = _merge_content_trees(
                source_git,
                base_tree=base_tree,
                local_tree=local_tree,
                upstream_tree=upstream_tree,
                message=f"Integrate frozen upstream changes for {source.display_name}",
            )
        except subprocess.CalledProcessError as error:
            if not source_git.unmerged_paths():
                raise
            unresolved = _apply_preserved_resolutions(source_git, conflict.root)
            if unresolved:
                source_git.reset_hard(conflict.local_commit)
                raise ValueError(
                    f"Conflict resolution is still required for: "
                    f"{', '.join(unresolved)}. Edit the files in {conflict.root}, "
                    f"then retry: {_conflict_retry_command(source, conflict.kind)}"
                ) from error
            source_git.run(
                [
                    "commit",
                    "--no-edit",
                    "-m",
                    f"Resolve frozen upstream conflict for {source.display_name}",
                ]
            )
            result_tree = source_git.tree("HEAD")
            if result_tree is None:
                raise ValueError(
                    f"Could not read the resolved content for {source.display_name}."
                )
        parsed_source = _finish_content_merge(
            source,
            source_git,
            local_commit=conflict.local_commit,
            result_tree=result_tree,
            upstream_commit=conflict.upstream_commit,
        )
    except BaseException:
        source_git.reset_hard(conflict.local_commit)
        raise

    _clear_conflict_state(source_git, conflict.root)
    upstream_ref = _upstream_ref(source_git)
    newer_upstream = False
    try:
        source_git.fetch("upstream")
        current_upstream = source_git.ref_sha(upstream_ref)
        newer_upstream = bool(
            current_upstream and current_upstream != conflict.upstream_commit
        )
    except subprocess.CalledProcessError as error:
        logger.debug(
            "Could not check for a newer update after resolving %s: %s",
            source.display_name,
            error,
        )
    return _IntegrationResult(
        saved_commit=None,
        parsed_source=parsed_source,
        newer_upstream=newer_upstream,
        resolved_conflict=True,
    )


def _integrate_upstream(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
    *,
    kind: str,
    submission_accepted: bool = False,
    requested_title: str | None = None,
) -> _IntegrationResult:
    conflict = _load_conflict_state(collection_root, source, source_git)
    if conflict is not None:
        if conflict.kind != kind:
            raise ValueError(
                f"{source.display_name} has an unresolved {conflict.kind} conflict. "
                f"Retry it with: {_conflict_retry_command(source, conflict.kind)}"
            )
        return _resolve_frozen_conflict(source, source_git, conflict)

    validate_collab_checkout(source_git, display_name=source.display_name)
    saved_commit = source_git.checkpoint(
        f"Save local deck changes for {source.display_name}"
    )
    local_commit = source_git.head()
    if local_commit is None:
        raise ValueError(f"Collab source {source.display_name} has no commits.")
    local_tree = source_git.tree(local_commit)
    if local_tree is None:
        raise ValueError(
            f"Could not read the local checkpoint for {source.display_name}."
        )

    try:
        source_git.fetch("upstream")
    except subprocess.CalledProcessError as error:
        checkpoint = (
            f" Local contribution committed at {local_commit[:12]}."
            if saved_commit
            else ""
        )
        raise ValueError(
            f"Could not fetch GitHub updates for {source.display_name}."
            f"{checkpoint} Retry exactly: {_conflict_retry_command(source, kind)}"
        ) from error
    upstream_ref = _upstream_ref(source_git)
    upstream_commit = source_git.ref_sha(upstream_ref)
    upstream_tree = source_git.tree(upstream_ref)
    if upstream_commit is None or upstream_tree is None:
        raise ValueError(
            f"Could not read the available update for {source.display_name}."
        )
    validate_collab_checkout(
        source_git,
        display_name=source.display_name,
        ref=upstream_ref,
    )

    base_ref = SUBMISSION_REF if submission_accepted else INTEGRATED_REF
    base_commit = source_git.ref_sha(base_ref)
    base_tree = source_git.tree(base_ref)
    if base_commit is None or base_tree is None:
        raise ValueError(
            f"Could not read the integrated checkpoint for {source.display_name}."
        )
    try:
        result_tree = _merge_content_trees(
            source_git,
            base_tree=base_tree,
            local_tree=local_tree,
            upstream_tree=upstream_tree,
            message=f"Integrate upstream changes for {source.display_name}",
        )
    except subprocess.CalledProcessError as error:
        if not source_git.unmerged_paths():
            source_git.reset_hard(local_commit)
            raise
        paths = source_git.unmerged_paths()
        source_git.update_refs(
            {
                CONFLICT_BASE_REF: base_commit,
                CONFLICT_LOCAL_REF: local_commit,
                CONFLICT_UPSTREAM_REF: upstream_commit,
            }
        )
        conflict_root = _save_conflict_copies(
            collection_root,
            source,
            source_git,
            kind=kind,
            requested_title=requested_title,
        )
        source_git.reset_hard(local_commit)
        raise ValueError(
            f"Local contribution committed at {local_commit[:12]}. Local and "
            f"upstream edits overlap in: {', '.join(paths)}. Resolve the editable "
            f"files in {conflict_root}, then retry exactly: "
            f"{_conflict_retry_command(source, kind)}"
        ) from error
    except BaseException:
        source_git.reset_hard(local_commit)
        raise

    try:
        parsed_source = _finish_content_merge(
            source,
            source_git,
            local_commit=local_commit,
            result_tree=result_tree,
            upstream_commit=upstream_commit,
        )
    except BaseException:
        source_git.reset_hard(local_commit)
        raise
    return _IntegrationResult(
        saved_commit=saved_commit,
        parsed_source=parsed_source,
    )


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
        source_git.copy_identity_from(collection_git)
        default_branch = source_git.default_branch("upstream")
        upstream_ref = f"upstream/{default_branch}"
        source_git.checkout_or_create_branch(JOURNAL_BRANCH, upstream_ref)
        source_git.run(["branch", "--unset-upstream"], check=False)
        source_git.update_ref(INTEGRATED_REF, upstream_ref)
        validate_collab_checkout(source_git, display_name=source.display_name)
        parsed_source = parse_source(source)
        require_note_keys(deck.parsed for deck in parsed_source.decks)
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
    retry_command = shlex.join(
        ["ankiops", "collab", "publish", args.deck, source.display_name]
    )
    try:
        publish_collab_deck(collection_root, args.deck, source)
    except RepositoryCollisionError as error:
        raise ValueError(str(error)) from error
    except Exception as error:
        source_git = GitRepository(source.root)
        prepared = (
            prepared_publish(source_git)
            if source.root.exists() and source_git.is_repo()
            else None
        )
        prepared_detail = (
            f"Prepared repository: {source.root}. " if prepared is not None else ""
        )
        raise ValueError(
            f"Publishing did not finish for {source.display_name}. {prepared_detail}"
            f"Retry exactly: {retry_command}. "
            f"Details: {error}"
        ) from error
    github_url = str(source.github_url).removesuffix(".git")
    _log_result(
        "publish",
        source,
        f"published '{args.deck}'",
        details=[
            f"Local repository: {source.root}",
            f"GitHub repository: {github_url}",
        ],
        next_steps=[],
    )


def _github_slug_from_remote(url: str) -> str | None:
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
        r"([^/]+)/([^/]+?)(?:\.git)?/?",
        url,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return parse_github_slug(f"{match.group(1)}/{match.group(2)}")
    except ValueError:
        return None


def _submission_state(
    source: DeckSource,
    source_git: GitRepository,
    github: GitHubHost,
) -> _SubmissionState:
    snapshot = source_git.ref_sha(SUBMISSION_REF)
    if snapshot is None:
        return _SubmissionState(_SubmissionPhase.NONE, None, None, None)

    publish_slug = _github_slug_from_remote(source_git.remote_url("publish") or "")
    if publish_slug is None:
        return _SubmissionState(
            _SubmissionPhase.PREPARED,
            snapshot,
            None,
            None,
        )

    owner = publish_slug.split("/", 1)[0]
    pull_request = github.find_pull_request(
        source.display_name,
        f"{owner}:{CONTRIBUTION_BRANCH}",
    )
    if (
        pull_request is not None
        and pull_request.state in {"MERGED", "CLOSED"}
        and pull_request.head_sha != snapshot
    ):
        pull_request = None
    if pull_request is not None and pull_request.state == "MERGED":
        phase = _SubmissionPhase.MERGED
    elif pull_request is not None and pull_request.state == "CLOSED":
        phase = _SubmissionPhase.CLOSED
    elif pull_request is not None and pull_request.state == "OPEN":
        phase = _SubmissionPhase.PR_OPEN
    else:
        phase = _SubmissionPhase.PREPARED
    return _SubmissionState(
        phase,
        snapshot,
        publish_slug,
        pull_request,
    )


def _update_one(collection_root: Path, source: DeckSource) -> None:
    source_git = _require_source_git(source)
    anki_paths_before = parse_source(source).applicable_paths
    integrated_before = source_git.ref_sha(INTEGRATED_REF)
    submission: _SubmissionState | None = None
    if source_git.ref_sha(SUBMISSION_REF):
        try:
            submission = _submission_state(
                source,
                source_git,
                GitHubHost(source.root),
            )
        except (ValueError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"Could not inspect the active submission for "
                f"{source.display_name}, so its update was not started. Retry "
                f"exactly: ankiops collab update {source.display_name}. "
                f"Details: {error}"
            ) from error
    result = _integrate_upstream(
        collection_root,
        source,
        source_git,
        kind="update",
        submission_accepted=bool(
            submission and submission.phase == _SubmissionPhase.MERGED
        ),
    )
    upstream_changed = bool(
        integrated_before
        and source_git.tree(integrated_before) != source_git.tree(INTEGRATED_REF)
    )
    changed_paths: list[GitPathChange] = []
    anki_applicable_changes = False
    if upstream_changed and integrated_before:
        changed_paths = source_git.diff_name_status(integrated_before, INTEGRATED_REF)
        flat_changed_paths = {
            path for changed_path in changed_paths for path in changed_path.paths
        }
        anki_applicable_changes = bool(
            flat_changed_paths
            & (anki_paths_before | result.parsed_source.applicable_paths)
        )
    local_contribution = source_git.tree("HEAD") != source_git.tree(INTEGRATED_REF)
    if submission and submission.phase in {
        _SubmissionPhase.MERGED,
        _SubmissionPhase.CLOSED,
    }:
        source_git.delete_ref(SUBMISSION_REF)

    if result.resolved_conflict:
        summary = "resolved frozen conflict and integrated upstream changes"
    elif upstream_changed and result.saved_commit:
        summary = "committed local changes and integrated upstream changes"
    elif upstream_changed:
        summary = "integrated upstream changes"
    elif result.saved_commit:
        summary = "committed local changes; upstream already current"
    elif local_contribution:
        summary = "no upstream changes"
    else:
        summary = "already up to date"

    details = []
    if upstream_changed:
        details.append(f"Files changed: {_counted(len(changed_paths), 'path')}")
    if anki_applicable_changes:
        details.append("Anki: changes available")
    if local_contribution:
        details.append("Local contribution: ready to submit")

    next_steps = []
    if result.newer_upstream:
        next_steps.append(
            f"Integrate the newer update: ankiops collab update {source.display_name}"
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
    source = _collab_source(collection_root, args.repository)
    if not source.root.exists():
        raise ValueError(f"Unknown collab source: {source.display_name}")
    _update_one(collection_root, source)


def run_submit(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    source = _collab_source(collection_root, args.repository)
    source_git = _require_source_git(source)
    parsed_source = parse_source(source)
    require_note_keys(deck.parsed for deck in parsed_source.decks)
    requested_title = getattr(args, "title", None)
    conflict = _load_conflict_state(collection_root, source, source_git)
    if conflict is not None:
        if conflict.kind != "submit":
            raise ValueError(
                f"An update conflict for {source.display_name} still needs "
                "resolution. Retry exactly: "
                f"{_conflict_retry_command(source, 'update')}"
            )
        if requested_title is None:
            requested_title = conflict.requested_title

    github = GitHubHost(source.root)
    try:
        submission = _submission_state(source, source_git, github)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"Could not inspect the durable submission state for "
            f"{source.display_name}. Nothing was sent. Retry exactly: "
            f"ankiops collab submit {source.display_name}. Details: {error}"
        ) from error
    if submission.phase == _SubmissionPhase.MERGED:
        _log_result(
            "submit",
            source,
            "pull request merged",
            details=[f"Pull request: {submission.pull_request.url}"],
            next_steps=[
                f"Integrate the merged contribution: ankiops collab update "
                f"{source.display_name}"
            ],
        )
        return
    if submission.phase == _SubmissionPhase.CLOSED:
        source_git.delete_ref(SUBMISSION_REF)
        submission = _SubmissionState(_SubmissionPhase.NONE, None, None, None)

    snapshot_tree = (
        source_git.tree(submission.snapshot) if submission.snapshot else None
    )
    draft_changed = bool(
        submission.snapshot
        and (source_git.status_lines() or source_git.tree("HEAD") != snapshot_tree)
    )
    prepare_snapshot = (
        submission.snapshot is None or draft_changed or bool(requested_title)
    )
    stored_title = None
    if submission.snapshot:
        message = source_git.commit_message(submission.snapshot)
        if message:
            stored_title = message.splitlines()[0]
    prepared_new_snapshot = False
    if prepare_snapshot:
        result = _integrate_upstream(
            collection_root,
            source,
            source_git,
            kind="submit",
            requested_title=requested_title,
        )
        upstream_parent = source_git.ref_sha(INTEGRATED_REF)
        local_tree = source_git.tree("HEAD")
        upstream_tree = source_git.tree(INTEGRATED_REF)
        if upstream_parent is None or local_tree is None or upstream_tree is None:
            raise ValueError(
                f"Could not prepare the contribution for {source.display_name}."
            )
        if local_tree == upstream_tree:
            source_git.delete_ref(SUBMISSION_REF)
            _log_result(
                "submit",
                source,
                (
                    "resolved the frozen conflict; nothing to submit"
                    if result.resolved_conflict
                    else "nothing to submit"
                ),
                details=(
                    ["Local changes: committed; content matches upstream"]
                    if result.saved_commit
                    else None
                ),
                next_steps=(
                    [
                        f"Integrate the newer update: ankiops collab update "
                        f"{source.display_name}"
                    ]
                    if result.newer_upstream
                    else None
                ),
            )
            return
        title = (
            requested_title
            or stored_title
            or _derive_submit_title(source_git, upstream_parent)
        )
        snapshot = source_git.create_commit(local_tree, upstream_parent, title)
        source_git.update_ref(SUBMISSION_REF, snapshot)
        prepared_new_snapshot = True
        if result.newer_upstream:
            _log_result(
                "submit",
                source,
                "resolved the frozen conflict; submission prepared locally",
                details=[
                    f"Commit: {title}",
                    "Upstream: a newer update is available",
                ],
                next_steps=[
                    f"Integrate the newer update: ankiops collab update "
                    f"{source.display_name}"
                ],
            )
            return
    else:
        snapshot = str(submission.snapshot)
        if not stored_title:
            raise ValueError(
                f"The prepared contribution for {source.display_name} has no title."
            )
        title = stored_title

    contribution_slug = submission.publish_slug
    if contribution_slug is None:
        try:
            contribution_slug, contributor = github.publish_target(source.display_name)
        except ValueError as error:
            raise ValueError(
                f"Submission prepared locally for {source.display_name}, but GitHub "
                "setup did not finish. Retry exactly: ankiops collab submit "
                f"{source.display_name}. Details: {error}"
            ) from error
        source_git.set_remote(
            "publish",
            f"https://github.com/{contribution_slug}.git",
        )
    else:
        contributor = contribution_slug.split("/", 1)[0]

    try:
        source_git.push_force("publish", snapshot, CONTRIBUTION_BRANCH)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Submission prepared locally for {source.display_name}, but the push "
            "did not finish. Retry exactly: ankiops collab submit "
            f"{source.display_name}."
        ) from error

    contribution_head = f"{contributor}:{CONTRIBUTION_BRANCH}"
    try:
        pull_request = submission.pull_request
        if pull_request is not None and pull_request.state == "OPEN":
            pr_url = pull_request.url
            head_changed = pull_request.head_sha != snapshot
            title_changed = pull_request.title != title
            if title_changed:
                github.update_pr(pr_url, title=title)
            summary = (
                "pull request updated"
                if prepared_new_snapshot or head_changed or title_changed
                else "pull request open"
            )
        else:
            pr_url = github.create_pr(
                source.display_name,
                head=contribution_head,
                base=source_git.default_branch("upstream"),
                title=title,
                body=f"Submitted with AnkiOps from {source.display_name}.",
            )
            summary = "pull request opened"
    except ValueError as error:
        raise ValueError(
            f"Submission pushed for {source.display_name}, but the pull request was "
            "not confirmed. Retry exactly: ankiops collab submit "
            f"{source.display_name}. Details: {error}"
        ) from error
    _log_result(
        "submit",
        source,
        summary,
        details=[
            f"Commit: {title}",
            f"GitHub repository: https://github.com/{contribution_slug}",
            f"Pull request: {pr_url}",
        ],
    )


def _repository_state(source_git: GitRepository) -> _RepositoryState:
    upstream_ref = _upstream_ref(source_git)
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


def _status_one(source: DeckSource) -> None:
    source_git = GitRepository(source.root)
    source_git.ensure_repo(
        f"The collab source {source.display_name} is not a valid repository."
    )
    publish_preparation = prepared_publish(source_git)
    if publish_preparation is not None:
        output_line(source.display_name)
        output_line("  Publish: prepared; handoff incomplete")
        output_line(f"  Local repository: {source.root}")
        _log_next_steps(
            [
                "Retry publish: "
                + shlex.join(
                    [
                        "ankiops",
                        "collab",
                        "publish",
                        publish_preparation.deck,
                        source.display_name,
                    ]
                )
            ]
        )
        return
    source_git = _require_source_git(source)

    local_changes = source_git.status_lines()
    conflict = _load_conflict_state(source.collection_root, source, source_git)
    repository_state: _RepositoryState | None = None
    submission: _SubmissionState | None = None
    try:
        source_git.fetch("upstream")
        repository_state = _repository_state(source_git)
    except (ValueError, subprocess.CalledProcessError) as error:
        logger.debug("Could not check GitHub for %s: %s", source.display_name, error)
    try:
        submission = _submission_state(
            source,
            source_git,
            GitHubHost(source.root),
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        logger.debug(
            "Could not read submission state for %s: %s",
            source.display_name,
            error,
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
        contribution_state = (
            "in pull request"
            if submission and submission.phase == _SubmissionPhase.PR_OPEN
            else "ready to submit"
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
        contribution_state = (
            "in pull request"
            if submission and submission.phase == _SubmissionPhase.PR_OPEN
            else "ready to submit"
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
    if conflict:
        output_line(f"  Conflict: frozen {conflict.kind} at {conflict.root}")
    if submission is None:
        if source_git.ref_sha(SUBMISSION_REF):
            output_line("  Submission: GitHub state unavailable")
    elif submission.phase != _SubmissionPhase.NONE:
        labels = {
            _SubmissionPhase.PREPARED: "prepared locally",
            _SubmissionPhase.PR_OPEN: "pull request open",
            _SubmissionPhase.MERGED: "merged; update to integrate",
            _SubmissionPhase.CLOSED: "pull request closed without merge",
        }
        output_line(f"  Submission: {labels[submission.phase]}")
        if submission.pull_request:
            output_line(f"  Pull request: {submission.pull_request.url}")

    next_steps = []
    if conflict:
        next_steps.append(
            f"Resolve and retry: {_conflict_retry_command(source, conflict.kind)}"
        )
    elif submission and submission.phase == _SubmissionPhase.MERGED:
        next_steps.append(
            f"Integrate merge: ankiops collab update {source.display_name}"
        )
    elif repository_state and repository_state.relation in {
        _RepositoryRelation.BEHIND,
        _RepositoryRelation.DIVERGED,
    }:
        next_steps.append(
            f"Integrate upstream: ankiops collab update {source.display_name}"
        )
    elif submission is None and source_git.ref_sha(SUBMISSION_REF):
        next_steps.append(f"Retry status: ankiops collab status {source.display_name}")
    elif submission and submission.phase == _SubmissionPhase.CLOSED:
        next_steps.append(f"Submit again: ankiops collab submit {source.display_name}")
    elif submission and submission.phase == _SubmissionPhase.PREPARED:
        next_steps.append(
            f"Retry submission: ankiops collab submit {source.display_name}"
        )
    elif (
        submission
        and submission.phase == _SubmissionPhase.PR_OPEN
        and (
            local_changes or source_git.tree("HEAD") != source_git.tree(SUBMISSION_REF)
        )
    ):
        next_steps.append(
            f"Update pull request: ankiops collab submit {source.display_name}"
        )
    elif submission and submission.phase == _SubmissionPhase.PR_OPEN:
        pass
    elif local_changes or (
        repository_state and repository_state.relation is _RepositoryRelation.AHEAD
    ):
        next_steps.append(
            f"Submit contribution: ankiops collab submit {source.display_name}"
        )
    elif repository_state is None:
        next_steps.append(f"Retry status: ankiops collab status {source.display_name}")
    _log_next_steps(next_steps)


def run_status(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    sources = discover_deck_sources(collection_root)[1:]
    requested_source: DeckSource | None = None
    if getattr(args, "repository", None):
        requested_source = _collab_source(collection_root, args.repository)
        sources = [requested_source] if requested_source.root.exists() else []
    if requested_source is not None and not requested_source.root.exists():
        raise ValueError(f"Unknown collab source: {requested_source.display_name}")
    if not sources:
        output_line("No subscribed collab decks.")
        _log_next_steps(["Subscribe to a deck: ankiops collab subscribe OWNER/REPO"])
        return
    for index, source in enumerate(sources):
        if index:
            output_line()
        _status_one(source)


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

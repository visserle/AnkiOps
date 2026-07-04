"""Independent-repository collab source commands."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from ankiops.collab.errors import format_missing_note_keys_error
from ankiops.collab.hosting import GitHubHost
from ankiops.collab.publish import publish_collab_deck
from ankiops.collection import require_collection_root
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
    source_content_hash,
)
from ankiops.git import GitRepository
from ankiops.markdown import read_deck_file
from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)

WORK_BRANCH = "ankiops/work"
_SAFE_SLUG_PART_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")


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
    if not all(_SAFE_SLUG_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(
            f"Invalid collab deck identity '{repository}': owner and repository "
            "may use ASCII letters, digits, and hyphens."
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
    for md_file in source.deck_files():
        parsed = read_deck_file(md_file, note_types=configs, context_root=source.root)
        for index, note in enumerate(parsed.notes, start=1):
            if not note.note_key:
                missing.append(f"{md_file.name} note {index}")
    if missing:
        raise ValueError(format_missing_note_keys_error(len(missing)))


def _log_result(
    action: str,
    source: DeckSource,
    summary: str,
    *,
    details: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> None:
    logger.info("Collab %s: %s — %s", action, source.display_name, summary)
    for detail in details or []:
        logger.info("  %s", detail)
    _log_next_steps(next_steps or [])


def _log_next_steps(steps: list[str]) -> None:
    if not steps:
        return
    logger.info("Next:")
    for step in steps:
        logger.info("  %s", step)


def _source_is_applied(source: DeckSource, state: SyncState) -> bool:
    applied_state = state.get_source_applied_state(source.source_path)
    return bool(applied_state and applied_state[0] == source_content_hash(source))


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


def _save_conflict_copies(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
    operation_id: str,
) -> Path:
    owner, repo_name = source.display_name.split("/", 1)
    conflict_root = (
        collection_root / ".ankiops" / "conflicts" / owner / repo_name / operation_id
    )
    for rel_path in source_git.unmerged_paths():
        versions: dict[str, bytes] = {}
        for stage, label in ((1, "base"), (2, "local"), (3, "upstream")):
            result = subprocess.run(
                ["git", "show", f":{stage}:{rel_path}"],
                cwd=source_git.root,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                continue
            versions[label] = result.stdout
            target = conflict_root / f"{rel_path}.{label}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.stdout)
        editable = conflict_root / rel_path
        editable.parent.mkdir(parents=True, exist_ok=True)
        merged = source_git.root / rel_path
        if merged.exists():
            editable.write_bytes(merged.read_bytes())
        else:
            editable.write_bytes(versions.get("local", versions.get("upstream", b"")))
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


def _has_conflict_markers(path: Path) -> bool:
    if not path.exists():
        return False
    content = path.read_bytes()
    return any(marker in content for marker in (b"<<<<<<<", b"=======", b">>>>>>>"))


def _apply_preserved_resolutions(
    transaction_git: GitRepository, conflict_root: Path
) -> list[str]:
    unresolved = []
    for rel_path in transaction_git.unmerged_paths():
        resolution = conflict_root / rel_path
        if not resolution.exists() or _has_conflict_markers(resolution):
            unresolved.append(rel_path)
            continue
        target = transaction_git.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolution, target)
        transaction_git.run(["add", "--", rel_path])
    return unresolved


def _integrate_upstream(
    collection_root: Path,
    source: DeckSource,
    source_git: GitRepository,
    state: SyncState,
    *,
    kind: str,
) -> tuple[str, bool, str | None, str, str]:
    operation_id, existing = _source_operation(state, source, kind)
    original_commit = source_git.head()
    if original_commit is None:
        raise ValueError(f"Collab source {source.display_name} has no commits.")
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
            if existing.get("recovery_ref"):
                shutil.rmtree(str(existing["recovery_ref"]), ignore_errors=True)
            return (
                operation_id,
                prepared_head != expected_head,
                None,
                str(upstream_tree),
                source_git.default_branch("upstream"),
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
        if transaction_git.trees_equal("HEAD", upstream_ref):
            files_changed = transaction_git.head() != original_commit
        else:
            files_changed = (
                transaction_git.integrate(
                    upstream_ref, f"Merge upstream changes for {source.display_name}"
                )
                or transaction_git.head() != original_commit
            )
    except subprocess.CalledProcessError as error:
        if "transaction_git" in locals() and transaction_git.unmerged_paths():
            if existing and existing.get("recovery_ref"):
                conflict_root = Path(str(existing["recovery_ref"]))
                unresolved = _apply_preserved_resolutions(
                    transaction_git, conflict_root
                )
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
                    upstream_tree=None,
                    recovery_ref=str(conflict_root),
                    last_error=str(error),
                )
                shutil.rmtree(transaction_path.parent, ignore_errors=True)
                raise ValueError(
                    f"Local and upstream edits overlap in: {paths}. The subscribed "
                    f"deck was not changed. Edit the preserved Markdown in "
                    f"{conflict_root}, remove its conflict markers, then retry: "
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

    transaction_commit = transaction_git.head()
    if transaction_commit is None:
        raise ValueError(f"Could not prepare the update for {source.display_name}.")
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
    source_git.run(["fetch", str(transaction_git.root), transaction_commit])
    source_git.reset_hard("FETCH_HEAD")
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
    _require_collection_git(collection_root)
    source = _collab_source(collection_root, args.repository)
    if source.root.exists():
        raise ValueError(f"Collab source already exists: {source.display_name}")
    github = GitHubHost(collection_root)
    github.ensure_authenticated()
    if github.repo_info(source.display_name) is None:
        raise ValueError(f"GitHub repository is not accessible: {source.display_name}")
    try:
        source_git = GitRepository.clone(str(source.github_url), source.root)
        default_branch = source_git.default_branch("upstream")
        source_git.checkout_or_create_branch(WORK_BRANCH, f"upstream/{default_branch}")
        _ensure_submittable_note_keys(source)
    except Exception:
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
    operation_id, _existing = _source_operation(state, source, "publish")
    state.save_collab_operation(
        source.source_path,
        operation_id,
        "publish",
        "publishing",
    )
    try:
        publish_collab_deck(
            collection_root,
            args.deck,
            source,
        )
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
            f"ankiops collab publish {args.deck!r} {source.display_name}. "
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
    )


def _update_one(collection_root: Path, source: DeckSource, state: SyncState) -> None:
    source_git = _require_source_git(source)
    before_commit = source_git.head()
    pending_action = state.get_collab_operation(source.source_path)
    operation_kind = str(pending_action["kind"]) if pending_action else "update"
    post_submission_states = {"ready", "push_failed", "pushed", "pr_failed", "pr_open"}
    try:
        _operation_id, files_changed, saved_commit, upstream_tree, _default_branch = (
            _integrate_upstream(
                collection_root, source, source_git, state, kind=operation_kind
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
    after_commit = source_git.head()
    upstream_changed = bool(
        after_commit and after_commit != (saved_commit or before_commit)
    )
    changed_paths: list[str] = []
    if files_changed and before_commit and after_commit:
        changed_paths = source_git.run(
            ["diff", "--name-status", before_commit, after_commit]
        ).stdout.splitlines()
    local_contribution = source_git.tree("HEAD") != upstream_tree
    if (
        pending_action
        and pending_action["kind"] == "submit"
        and pending_action["state"] in post_submission_states
    ):
        if source_git.tree("HEAD") == upstream_tree:
            if pending_action.get("publish_branch") and source_git.remote_url(
                "publish"
            ):
                source_git.delete_remote_branch(
                    "publish", str(pending_action["publish_branch"])
                )
            state.clear_collab_operation(source.source_path)
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

    next_steps = []
    if not _source_is_applied(source, state):
        next_steps.append("Apply to Anki: ankiops fa")
    if remaining_action and remaining_action.get("pr_url"):
        next_steps.append(f"Review pull request: {remaining_action['pr_url']}")
    elif remaining_action and remaining_action["kind"] == "submit":
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
        logger.info("Collab update: no subscribed decks")
        _log_next_steps(["Subscribe to a deck: ankiops collab subscribe OWNER/REPO"])
        return
    state = SyncState.open(collection_root)
    failures = []
    try:
        for source in sources:
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
    title = (
        getattr(args, "message", None) or f"Update collab deck {source.display_name}"
    )
    state = SyncState.open(collection_root)
    try:
        pending_action = state.get_collab_operation(source.source_path)
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
        if (
            pending_action
            and pending_action["kind"] == "submit"
            and pending_action.get("pr_url")
        ):
            pr_url = str(pending_action["pr_url"])
            _log_result(
                "submit",
                source,
                "pull request already open",
                details=[f"Pull request: {pr_url}"],
                next_steps=[f"Review pull request: {pr_url}"],
            )
            return

        saved_commit: str | None = None
        if pending_action is None or pending_action["state"] in {
            "integrating",
            "applying",
        }:
            (
                operation_id,
                _files_changed,
                saved_commit,
                upstream_tree,
                target_branch,
            ) = _integrate_upstream(
                collection_root, source, source_git, state, kind="submit"
            )
            if source_git.tree("HEAD") == upstream_tree:
                next_steps = []
                if not _source_is_applied(source, state):
                    next_steps.append("Apply to Anki: ankiops fa")
                _log_result(
                    "submit",
                    source,
                    "nothing to submit",
                    details=(
                        ["Local changes: committed; content matches upstream"]
                        if saved_commit
                        else None
                    ),
                    next_steps=next_steps,
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
                pushed_sha=None,
                pr_url=None,
                last_error=None,
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
        github = GitHubHost(source.root)
        try:
            contribution_slug, contributor = github.publish_target(source.display_name)
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
        source_git.set_remote("publish", f"https://github.com/{contribution_slug}.git")
        uploaded_commit = source_git.remote_branch_sha("publish", contribution_branch)
        if uploaded_commit != local_commit:
            try:
                source_git.push("publish", "HEAD", contribution_branch)
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
            pushed_sha=local_commit,
            last_error=None,
        )
        contribution_head = f"{contributor}:{contribution_branch}"
        try:
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
                pushed_sha=local_commit,
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
            pushed_sha=local_commit,
            pr_url=pr_url,
            last_error=None,
        )
        _log_result(
            "submit",
            source,
            "pull request opened",
            details=[
                (
                    "Local contribution: committed and uploaded"
                    if saved_commit
                    else "Local contribution: uploaded existing commit"
                ),
                f"GitHub repository: https://github.com/{contribution_slug}",
                f"Pull request: {pr_url}",
            ],
            next_steps=[f"Review and merge: {pr_url}"],
        )
    finally:
        state.close()


def _repository_state(source_git: GitRepository) -> _RepositoryState:
    _default_branch, upstream_ref = _upstream_ref(source_git)
    if source_git.trees_equal("HEAD", upstream_ref):
        return _RepositoryState(_RepositoryRelation.CURRENT, 0, 0)

    counts = source_git.run(
        ["rev-list", "--left-right", "--count", f"{upstream_ref}...HEAD"]
    ).stdout.split()
    upstream_commits, local_commits = (int(value) for value in counts)
    if local_commits and upstream_commits:
        relation = _RepositoryRelation.DIVERGED
    elif local_commits:
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
    is_applied = _source_is_applied(source, state)
    submission_accepted = bool(
        not local_changes
        and repository_state
        and repository_state.relation is _RepositoryRelation.CURRENT
        and pending_action
        and pending_action["kind"] == "submit"
    )

    logger.info("Collab status: %s", source.display_name)
    working_tree = (
        "clean" if not local_changes else _counted(len(local_changes), "changed path")
    )
    logger.info("  Working tree: %s", working_tree)
    for line in local_changes:
        logger.info("    %s", line)

    if repository_state is None:
        logger.info("  Local contribution: unknown")
        logger.info("  Upstream: unavailable")
    elif repository_state.relation is _RepositoryRelation.CURRENT:
        logger.info("  Local contribution: none")
        logger.info("  Upstream: up to date")
    elif repository_state.relation is _RepositoryRelation.AHEAD:
        logger.info(
            "  Local contribution: %s ready to submit",
            _counted(repository_state.local_commits, "commit"),
        )
        logger.info("  Upstream: up to date")
    elif repository_state.relation is _RepositoryRelation.BEHIND:
        logger.info("  Local contribution: none")
        logger.info(
            "  Upstream: %s available",
            _counted(repository_state.upstream_commits, "commit"),
        )
    else:
        logger.info(
            "  Local contribution: %s ready to submit",
            _counted(repository_state.local_commits, "commit"),
        )
        logger.info(
            "  Upstream: %s available; update before submitting",
            _counted(repository_state.upstream_commits, "commit"),
        )

    logger.info("  Anki: %s", "up to date" if is_applied else "changes not applied")
    if pending_action:
        if pending_action["state"] == "conflict":
            logger.info(
                "  Update: conflict requires resolution at %s",
                pending_action.get("recovery_ref"),
            )
        elif pending_action["kind"] == "submit":
            submission_state = (
                "accepted upstream; cleanup pending"
                if submission_accepted
                else {
                    "ready": "ready to upload",
                    "push_failed": "not uploaded",
                    "pushed": "uploaded; pull request not confirmed",
                    "pr_failed": "uploaded; pull request not created",
                    "pr_open": "pull request open",
                }.get(str(pending_action["state"]), "interrupted")
            )
            logger.info("  Submission: %s", submission_state)
        else:
            logger.info(
                "  Interrupted action: %s %s",
                pending_action["kind"],
                pending_action["state"],
            )
        if pending_action.get("pr_url"):
            logger.info("  Pull request: %s", pending_action["pr_url"])

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
    elif submission_accepted:
        if not is_applied:
            next_steps.append("Apply to Anki: ankiops fa")
        next_steps.append(
            f"Finalize submission: ankiops collab update {source.display_name}"
        )
    else:
        if not is_applied:
            next_steps.append("Apply to Anki: ankiops fa")
        if pending_action and pending_action.get("pr_url"):
            next_steps.append(f"Review pull request: {pending_action['pr_url']}")
        elif pending_action and pending_action["kind"] == "submit":
            next_steps.append(
                f"Retry submission: ankiops collab submit {source.display_name}"
            )
        elif local_changes or repository_state.relation is _RepositoryRelation.AHEAD:
            next_steps.append(
                f"Submit contribution: ankiops collab submit {source.display_name}"
            )
    _log_next_steps(next_steps)


def run_status(args: SimpleNamespace) -> None:
    collection_root = require_collection_root()
    _require_collection_git(collection_root)
    sources = discover_deck_sources(collection_root)[1:]
    if getattr(args, "repository", None):
        source = _collab_source(collection_root, args.repository)
        if not source.root.exists():
            raise ValueError(f"Unknown collab source: {source.display_name}")
        sources = [source]
    if not sources:
        logger.info("Collab status: no subscribed decks")
        _log_next_steps(["Subscribe to a deck: ankiops collab subscribe OWNER/REPO"])
        return
    state = SyncState.open(collection_root)
    try:
        for source in sources:
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

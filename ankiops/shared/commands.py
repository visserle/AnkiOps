"""Independent-repository shared source commands."""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from ankiops.collection import require_collection_dir
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
    source_content_hash,
)
from ankiops.git import RepositoryGit
from ankiops.markdown import read_deck_file
from ankiops.shared.create import create_shared_deck
from ankiops.shared.errors import format_missing_note_keys_error
from ankiops.shared.hosting import GitHubHost
from ankiops.sync.state import SyncState

logger = logging.getLogger(__name__)

WORK_BRANCH = "ankiops/work"
_SAFE_SLUG_PART_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")


def _parse_slug(slug: str) -> str:
    value = slug.strip().removesuffix(".git")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Expected GitHub repository as owner/repo")
    if not all(_SAFE_SLUG_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(
            f"Invalid GitHub repository '{slug}': owner and repository may use "
            "ASCII letters, digits, and hyphens."
        )
    return value


def _source_for_slug(collection_dir: Path, slug: str) -> DeckSource:
    return DeckSource.shared(collection_dir, _parse_slug(slug))


def _require_root_repo(collection_dir: Path) -> RepositoryGit:
    repo = RepositoryGit(collection_dir)
    repo.ensure_repo("AnkiOps collections require a Git repository at the root.")
    ignored = repo.run(
        ["check-ignore", "-q", "--no-index", "shared/.ankiops-probe"],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "This collection is not configured to keep shared decks separate. "
            "Add /shared/ to the root .gitignore, then retry the command."
        )
    if repo.run(["ls-files", "shared"], check=False).stdout.splitlines():
        raise ValueError(
            "The collection repository contains shared deck files from the old "
            "layout. This experimental version has no migration path; use a fresh "
            "collection."
        )
    return repo


def _require_source_repo(source: DeckSource) -> RepositoryGit:
    repo = RepositoryGit(source.root)
    repo.ensure_repo(
        f"The subscribed deck {source.source_id} is not a valid independent "
        f"repository at {source.root}. Leave the directory untouched for "
        "inspection and subscribe in a fresh collection path."
    )
    if repo.remote_url("upstream") is None:
        raise ValueError(
            f"The subscribed deck {source.source_id} has no GitHub source. "
            f"Subscribe to it again in a fresh collection path: ankiops shared "
            f"subscribe {source.source_id}"
        )
    return repo


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


def _log_outcome(
    source: DeckSource,
    *,
    files: str,
    local: str,
    github: str,
    retry: str,
    next_command: str,
    pr_url: str | None = None,
) -> None:
    logger.info("Shared deck: %s", source.source_id)
    logger.info("Shared files: %s", files)
    logger.info("Local result: %s", local)
    logger.info("Private decks: not touched")
    logger.info("GitHub: %s", github)
    if pr_url:
        logger.info("Pull request: %s", pr_url)
    logger.info("Safe to retry: %s", retry)
    logger.info("Next: %s", next_command)


def _source_operation(
    db: SyncState, source: DeckSource, kind: str
) -> tuple[str, dict[str, str | None] | None]:
    existing = db.get_shared_operation(source.source_id)
    if existing and existing["kind"] != kind:
        raise ValueError(
            f"{source.source_id} has an unfinished {existing['kind']} action. "
            f"Run ankiops shared status {source.source_id} for the exact next step."
        )
    operation_id = str(existing["operation_id"]) if existing else uuid4().hex[:12]
    return operation_id, existing


def _save_conflict_copies(
    collection_dir: Path,
    source: DeckSource,
    repo: RepositoryGit,
    operation_id: str,
) -> Path:
    owner, repo_name = source.source_id.split("/", 1)
    conflict_root = (
        collection_dir / ".ankiops" / "conflicts" / owner / repo_name / operation_id
    )
    for rel_path in repo.unmerged_paths():
        versions: dict[str, bytes] = {}
        for stage, label in ((1, "base"), (2, "local"), (3, "upstream")):
            result = subprocess.run(
                ["git", "show", f":{stage}:{rel_path}"],
                cwd=repo.root,
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
        merged = repo.root / rel_path
        if merged.exists():
            editable.write_bytes(merged.read_bytes())
        else:
            editable.write_bytes(versions.get("local", versions.get("upstream", b"")))
    return conflict_root


def _remote_ref(repo: RepositoryGit) -> tuple[str, str]:
    branch = repo.default_branch("upstream")
    return branch, f"upstream/{branch}"


def _worktree_fingerprint(repo: RepositoryGit) -> str:
    digest = hashlib.sha256()
    for args in (
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        ["diff", "--binary"],
        ["diff", "--cached", "--binary"],
    ):
        digest.update(repo.run(args).stdout.encode())
    for path in sorted(
        path
        for path in repo.root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ):
        digest.update(str(path.relative_to(repo.root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _copy_worktree(source: Path, target: Path) -> None:
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


def _copy_git_identity(source: RepositoryGit, target: RepositoryGit) -> None:
    for key in ("user.name", "user.email"):
        value = source.run(["config", "--get", key], check=False).stdout.strip()
        if value:
            target.run(["config", key, value])


def _has_conflict_markers(path: Path) -> bool:
    if not path.exists():
        return False
    content = path.read_bytes()
    return any(marker in content for marker in (b"<<<<<<<", b"=======", b">>>>>>>"))


def _apply_preserved_resolutions(
    transaction: RepositoryGit, conflict_root: Path
) -> list[str]:
    unresolved = []
    for rel_path in transaction.unmerged_paths():
        resolution = conflict_root / rel_path
        if not resolution.exists() or _has_conflict_markers(resolution):
            unresolved.append(rel_path)
            continue
        target = transaction.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolution, target)
        transaction.run(["add", "--", rel_path])
    return unresolved


def _integrate_upstream(
    collection_dir: Path,
    source: DeckSource,
    repo: RepositoryGit,
    db: SyncState,
    *,
    kind: str,
) -> tuple[bool, str | None, str, str]:
    operation_id, existing = _source_operation(db, source, kind)
    original_head = repo.head()
    if original_head is None:
        raise ValueError(f"Shared source {source.source_id} has no commits.")
    original_fingerprint = _worktree_fingerprint(repo)
    if (
        existing
        and existing.get("state") in {"fetching", "failed", "conflict"}
        and existing.get("head_commit")
        and existing["head_commit"] != original_fingerprint
    ):
        raise ValueError(
            f"The subscribed deck changed after its interrupted update. Its "
            f"files were left untouched. Review ankiops shared status "
            f"{source.source_id} before retrying."
        )
    db.save_shared_operation(
        source.source_id,
        operation_id,
        kind,
        "fetching",
        base_commit=original_head,
        head_commit=original_fingerprint,
        recovery_ref=str(existing.get("recovery_ref") or "") if existing else None,
    )
    transaction_parent = collection_dir / ".ankiops" / "transactions"
    transaction_parent.mkdir(parents=True, exist_ok=True)
    transaction_path = (
        Path(tempfile.mkdtemp(prefix=f"{operation_id}-", dir=transaction_parent))
        / "deck"
    )
    conflict_root: Path | None = None
    try:
        transaction = RepositoryGit.clone(
            str(source.root), transaction_path, remote="local"
        )
        _copy_git_identity(repo, transaction)
        _copy_worktree(source.root, transaction.root)
        checkpoint = transaction.checkpoint(
            f"Save local deck changes for {source.source_id}"
        )
        upstream_url = repo.remote_url("upstream")
        if upstream_url is None:
            raise ValueError(f"No GitHub source is configured for {source.source_id}.")
        transaction.run(["remote", "add", "upstream", upstream_url])
        transaction.fetch("upstream")
        branch, upstream_ref = _remote_ref(transaction)
        upstream_tree = transaction.tree(upstream_ref)
        if upstream_tree is None:
            raise ValueError(
                f"Could not read the available update for {source.source_id}."
            )
        if transaction.trees_equal("HEAD", upstream_ref):
            changed = transaction.head() != original_head
        else:
            changed = (
                transaction.integrate(
                    upstream_ref, f"Merge upstream changes for {source.source_id}"
                )
                or transaction.head() != original_head
            )
    except subprocess.CalledProcessError as error:
        if "transaction" in locals() and transaction.unmerged_paths():
            if existing and existing.get("recovery_ref"):
                conflict_root = Path(str(existing["recovery_ref"]))
                unresolved = _apply_preserved_resolutions(transaction, conflict_root)
                if not unresolved:
                    transaction.run(["commit", "--no-edit"])
                    changed = True
                    branch, upstream_ref = _remote_ref(transaction)
                    upstream_tree = transaction.tree(upstream_ref)
                    if upstream_tree is None:
                        raise ValueError(
                            "Could not read the available update for "
                            f"{source.source_id}."
                        )
                else:
                    conflict_root = Path(str(existing["recovery_ref"]))
            else:
                conflict_root = _save_conflict_copies(
                    collection_dir, source, transaction, operation_id
                )
            if transaction.unmerged_paths():
                paths = ", ".join(transaction.unmerged_paths())
                db.save_shared_operation(
                    source.source_id,
                    operation_id,
                    kind,
                    "conflict",
                    base_commit=original_head,
                    head_commit=original_fingerprint,
                    recovery_ref=str(conflict_root),
                    last_error=str(error),
                )
                shutil.rmtree(transaction_path.parent, ignore_errors=True)
                raise ValueError(
                    f"Local and upstream edits overlap in: {paths}. The subscribed "
                    f"deck was not changed. Edit the preserved Markdown in "
                    f"{conflict_root}, remove its conflict markers, then retry: "
                    f"ankiops shared update {source.source_id}"
                ) from error
        else:
            db.save_shared_operation(
                source.source_id,
                operation_id,
                kind,
                "failed",
                base_commit=original_head,
                head_commit=original_fingerprint,
                last_error=str(error),
            )
            shutil.rmtree(transaction_path.parent, ignore_errors=True)
            raise ValueError(
                f"GitHub could not be reached for {source.source_id}. The subscribed "
                f"deck was not changed and nothing was sent. Retrying is safe: "
                f"ankiops shared {kind} {source.source_id}"
            ) from error
    except ValueError as error:
        db.save_shared_operation(
            source.source_id,
            operation_id,
            kind,
            "failed",
            base_commit=original_head,
            head_commit=original_fingerprint,
            last_error=str(error),
        )
        shutil.rmtree(transaction_path.parent, ignore_errors=True)
        raise

    transaction_head = transaction.head()
    if transaction_head is None:
        raise ValueError(f"Could not prepare the update for {source.source_id}.")
    repo.run(["fetch", str(transaction.root), transaction_head])
    repo.reset_hard("FETCH_HEAD")
    shutil.rmtree(transaction_path.parent, ignore_errors=True)
    if conflict_root:
        shutil.rmtree(conflict_root, ignore_errors=True)
    db.clear_shared_operation(source.source_id)
    if upstream_tree is None:
        raise ValueError(f"Could not read the available update for {source.source_id}.")
    return changed, checkpoint, upstream_tree, branch


def run_subscribe(args: SimpleNamespace) -> None:
    collection_dir = require_collection_dir()
    _require_root_repo(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    if source.root.exists():
        raise ValueError(f"Shared source already exists: {source.source_id}")
    host = GitHubHost(collection_dir)
    host.ensure_authenticated()
    if host.repo_info(source.source_id) is None:
        raise ValueError(f"GitHub repository is not accessible: {source.source_id}")
    try:
        repo = RepositoryGit.clone(str(source.github_url), source.root)
        branch = repo.default_branch("upstream")
        repo.ensure_work_branch(WORK_BRANCH, f"upstream/{branch}")
        _ensure_submittable_note_keys(source)
    except Exception:
        if source.root.exists():
            shutil.rmtree(source.root)
        raise
    _log_outcome(
        source,
        files=f"subscribed at {source.root}",
        local="the shared Markdown is ready for review",
        github="downloaded the shared deck; nothing was sent",
        retry="not needed; subscription completed",
        next_command="ankiops fa",
    )


def run_publish(args: SimpleNamespace) -> None:
    collection_dir = require_collection_dir()
    _require_root_repo(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    db = SyncState.open(collection_dir)
    operation_id, _existing = _source_operation(db, source, "publish")
    db.save_shared_operation(
        source.source_id,
        operation_id,
        "publish",
        "publishing",
    )
    try:
        create_shared_deck(
            collection_dir,
            args.deck,
            source,
            public=bool(getattr(args, "public", False)),
        )
    except Exception as error:
        db.save_shared_operation(
            source.source_id,
            operation_id,
            "publish",
            "failed",
            last_error=str(error),
        )
        raise ValueError(
            f"Publishing did not finish for {source.source_id}. Your local deck "
            f"and any completed GitHub work were preserved. Retrying is safe: "
            f"ankiops shared publish {args.deck!r} {source.source_id}. "
            f"Details: {error}"
        ) from error
    else:
        db.clear_shared_operation(source.source_id)
    finally:
        db.close()
    _log_outcome(
        source,
        files=f"published deck '{args.deck}' from {source.root}",
        local="the deck is now an independent shared deck",
        github="created the repository and uploaded the deck",
        retry="not needed; publishing completed",
        next_command="ankiops fa",
    )


def _restore_operation(
    db: SyncState,
    source: DeckSource,
    operation: dict[str, str | None],
) -> None:
    db.save_shared_operation(
        source.source_id,
        str(operation["operation_id"]),
        str(operation["kind"]),
        str(operation["state"]),
        base_commit=operation.get("base_commit"),
        head_commit=operation.get("head_commit"),
        recovery_ref=operation.get("recovery_ref"),
        publish_branch=operation.get("publish_branch"),
        pushed_sha=operation.get("pushed_sha"),
        pr_url=operation.get("pr_url"),
        last_error=operation.get("last_error"),
    )


def _update_one(collection_dir: Path, source: DeckSource, db: SyncState) -> None:
    repo = _require_source_repo(source)
    before = repo.head()
    active = db.get_shared_operation(source.source_id)
    operation_kind = str(active["kind"]) if active else "update"
    changed, checkpoint, upstream_tree, _branch = _integrate_upstream(
        collection_dir, source, repo, db, kind=operation_kind
    )
    after = repo.head()
    if changed and before and after:
        files = repo.run(["diff", "--name-status", before, after]).stdout.splitlines()
        file_summary = f"{len(files)} path(s) updated"
    else:
        file_summary = "already current"
    if (
        active
        and active["kind"] == "submit"
        and active["state"] in {"push_failed", "pushed", "pr_failed", "pr_open"}
    ):
        if repo.tree("HEAD") == upstream_tree:
            if active.get("publish_branch") and repo.remote_url("publish"):
                repo.delete_remote_branch("publish", str(active["publish_branch"]))
        else:
            _restore_operation(db, source, active)
    _log_outcome(
        source,
        files=file_summary,
        local=(
            "saved local deck edits and applied available updates"
            if checkpoint
            else "applied available updates without changing private decks"
        ),
        github="checked for updates; nothing was sent",
        retry="yes; the update completed transactionally",
        next_command="ankiops fa",
    )


def run_update(args: SimpleNamespace) -> None:
    collection_dir = require_collection_dir()
    _require_root_repo(collection_dir)
    sources = discover_deck_sources(collection_dir)[1:]
    if getattr(args, "repo", None):
        requested = _source_for_slug(collection_dir, args.repo)
        if not requested.root.exists():
            raise ValueError(f"Unknown shared source: {requested.source_id}")
        sources = [requested]
    if not sources:
        logger.info("No shared sources found.")
        return
    db = SyncState.open(collection_dir)
    failures = []
    try:
        for source in sources:
            try:
                _update_one(collection_dir, source, db)
            except (ValueError, subprocess.CalledProcessError) as error:
                failures.append(f"{source.source_id}: {error}")
                logger.error("%s", failures[-1])
    finally:
        db.close()
    if failures:
        raise ValueError(
            f"Shared update finished with {len(failures)} failure(s): "
            + " | ".join(failures)
        )


def run_submit(args: SimpleNamespace) -> None:
    collection_dir = require_collection_dir()
    _require_root_repo(collection_dir)
    source = _source_for_slug(collection_dir, args.repo)
    repo = _require_source_repo(source)
    _ensure_submittable_note_keys(source)
    title = getattr(args, "message", None) or f"Update shared deck {source.source_id}"
    db = SyncState.open(collection_dir)
    try:
        active = db.get_shared_operation(source.source_id)
        if active and active["kind"] == "update":
            raise ValueError(
                f"An update for {source.source_id} still needs attention. Run: "
                f"ankiops shared update {source.source_id}"
            )
        if active and active["kind"] == "submit" and active.get("pr_url"):
            _log_outcome(
                source,
                files="the existing contribution is unchanged",
                local="no new contribution was created",
                github="the contribution and pull request already exist",
                retry="yes; retrying reuses the same pull request",
                next_command=f"review pull request {active['pr_url']}",
                pr_url=str(active["pr_url"]),
            )
            return

        checkpoint: str | None = None
        if active is None:
            _changed, checkpoint, upstream_tree, base = _integrate_upstream(
                collection_dir, source, repo, db, kind="submit"
            )
            if repo.tree("HEAD") == upstream_tree:
                _log_outcome(
                    source,
                    files="no contribution to submit",
                    local=(
                        "saved local deck edits; their content already matches GitHub"
                        if checkpoint
                        else "no local changes were committed"
                    ),
                    github="nothing was sent and no pull request was created",
                    retry="yes; this was a no-op",
                    next_command="none",
                )
                return
            operation_id, _existing = _source_operation(db, source, "submit")
            branch = f"ankiops/{operation_id}"
        else:
            if repo.status_lines() or repo.head() != active.get("head_commit"):
                raise ValueError(
                    f"A contribution for {source.source_id} is waiting to finish, "
                    "but the shared files changed afterward. Nothing was sent. "
                    f"Run: ankiops shared status {source.source_id}"
                )
            operation_id = str(active["operation_id"])
            branch = str(active["publish_branch"])
            upstream_tree = str(active["base_commit"])
            base = repo.default_branch("upstream")

        head = repo.head()
        if head is None:
            raise ValueError(f"Shared source {source.source_id} has no commits.")
        host = GitHubHost(source.root)
        publish_slug, head_owner = host.publish_target(source.source_id)
        repo.set_remote("publish", f"https://github.com/{publish_slug}.git")
        remote_sha = repo.remote_branch_sha("publish", branch)
        if remote_sha != head:
            try:
                repo.push("publish", "HEAD", branch)
            except subprocess.CalledProcessError as error:
                db.save_shared_operation(
                    source.source_id,
                    operation_id,
                    "submit",
                    "push_failed",
                    base_commit=upstream_tree,
                    head_commit=head,
                    publish_branch=branch,
                    last_error=str(error),
                )
                raise ValueError(
                    f"Your contribution for {source.source_id} was committed "
                    "locally, but it did not reach GitHub. Private decks were not "
                    "touched. Retrying is safe: ankiops shared submit "
                    f"{source.source_id}"
                ) from error

        db.save_shared_operation(
            source.source_id,
            operation_id,
            "submit",
            "pushed",
            base_commit=upstream_tree,
            head_commit=head,
            publish_branch=branch,
            pushed_sha=head,
        )
        pr_head = f"{head_owner}:{branch}"
        try:
            pr_url = host.create_pr(
                source.source_id,
                head=pr_head,
                base=base,
                title=title,
                body=f"Submitted with AnkiOps from {source.source_id}.",
            )
        except ValueError as error:
            db.save_shared_operation(
                source.source_id,
                operation_id,
                "submit",
                "pr_failed",
                base_commit=upstream_tree,
                head_commit=head,
                publish_branch=branch,
                pushed_sha=head,
                last_error=str(error),
            )
            raise ValueError(
                f"Your contribution reached GitHub, but its pull request was not "
                "created. No private files changed. Retrying is safe: "
                "ankiops shared submit "
                f"{source.source_id}. Details: {error}"
            ) from error
        db.save_shared_operation(
            source.source_id,
            operation_id,
            "submit",
            "pr_open",
            base_commit=upstream_tree,
            head_commit=head,
            publish_branch=branch,
            pushed_sha=head,
            pr_url=pr_url,
        )
        _log_outcome(
            source,
            files="shared deck changes submitted",
            local=(
                "committed changes belonging only to this shared deck"
                if checkpoint
                else "reused the existing local contribution"
            ),
            github=f"uploaded the contribution to {publish_slug}",
            retry="yes; retrying reuses this contribution and pull request",
            next_command=f"review pull request {pr_url}",
            pr_url=pr_url,
        )
    finally:
        db.close()


def _remote_state(repo: RepositoryGit) -> str:
    _branch, upstream_ref = _remote_ref(repo)
    if repo.trees_equal("HEAD", upstream_ref):
        return "current; no update is available"
    head = repo.head()
    upstream = repo.run(["rev-parse", upstream_ref]).stdout.strip()
    if head and repo.is_ancestor(head, upstream):
        return "an update is available"
    if head and repo.is_ancestor(upstream, head):
        return "local contributions have not been accepted upstream"
    return "both local contributions and upstream updates are present"


def _status_one(
    collection_dir: Path,
    source: DeckSource,
    db: SyncState,
    private_status: list[str],
) -> None:
    repo = _require_source_repo(source)
    local = repo.status_lines()
    try:
        repo.fetch("upstream")
        remote = _remote_state(repo)
    except (ValueError, subprocess.CalledProcessError) as error:
        logger.debug("Could not check GitHub for %s: %s", source.source_id, error)
        remote = "could not check GitHub; local files were not changed"
    operation = db.get_shared_operation(source.source_id)
    applied = db.get_source_applied_state(source.source_id)
    logger.info("Shared deck: %s", source.source_id)
    logger.info("Local shared changes: %d", len(local))
    for line in local:
        logger.info("  %s", line[3:] if len(line) > 3 else line)
    logger.info(
        "Private deck changes: %d (not part of shared commands)", len(private_status)
    )
    for line in private_status:
        logger.info("  %s", line[3:] if len(line) > 3 else line)
    logger.info("Available updates: %s", remote)
    logger.info(
        "Anki: %s",
        (
            "current source tree has been applied"
            if applied and applied[0] == source_content_hash(source)
            else "current source tree has not been applied; run ankiops fa"
        ),
    )
    if operation:
        if operation["state"] == "conflict":
            logger.info(
                "Interrupted update: the subscribed deck is unchanged; editable "
                "conflicts are preserved at %s",
                operation.get("recovery_ref"),
            )
        elif operation["kind"] == "submit":
            submission_state = {
                "push_failed": "not uploaded",
                "pushed": "uploaded; pull request not yet confirmed",
                "pr_failed": "uploaded; pull request creation did not finish",
                "pr_open": "pull request open",
            }.get(str(operation["state"]), "not finished")
            logger.info(
                "Pending submission: %s; retrying submit is safe",
                submission_state,
            )
        else:
            logger.info(
                "Previous action: did not finish; the shared deck was unchanged "
                "and retrying is safe"
            )
        if operation.get("pr_url"):
            logger.info("Pull request: %s", operation["pr_url"])
    if operation and operation["state"] == "conflict":
        next_command = f"ankiops shared update {source.source_id}"
    elif operation and operation["kind"] == "update":
        next_command = f"ankiops shared update {source.source_id}"
    elif operation and operation.get("pr_url"):
        next_command = f"review pull request {operation['pr_url']}"
    elif "available" in remote or "both local" in remote:
        next_command = f"ankiops shared update {source.source_id}"
    elif local or "unpublished" in remote:
        next_command = f"ankiops shared submit {source.source_id}"
    else:
        next_command = "none"
    logger.info("Next: %s", next_command)


def run_status(args: SimpleNamespace) -> None:
    collection_dir = require_collection_dir()
    root = _require_root_repo(collection_dir)
    private_status = root.status_lines()
    sources = discover_deck_sources(collection_dir)[1:]
    if getattr(args, "repo", None):
        source = _source_for_slug(collection_dir, args.repo)
        if not source.root.exists():
            raise ValueError(f"Unknown shared source: {source.source_id}")
        sources = [source]
    if not sources:
        logger.info("No shared deck subscriptions found.")
        logger.info("Next: ankiops shared subscribe OWNER/REPO")
        return
    db = SyncState.open(collection_dir)
    try:
        for source in sources:
            _status_one(collection_dir, source, db, private_status)
    finally:
        db.close()


def run(args: SimpleNamespace) -> None:
    handlers = {
        "publish": run_publish,
        "subscribe": run_subscribe,
        "update": run_update,
        "submit": run_submit,
        "status": run_status,
    }
    handler = handlers.get(args.shared_command)
    if handler is None:
        raise SystemExit(2)
    handler(args)

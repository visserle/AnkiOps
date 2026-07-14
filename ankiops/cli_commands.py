"""CLI command adapters and orchestration for argparse handlers."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from rich.markup import escape as rich_escape

from ankiops.collab import run as run_collab_impl
from ankiops.collab.source_security import validate_collection_collab_sources
from ankiops.collection import (
    LOCAL_MEDIA_DIR,
    NOTE_TYPES_DIR,
    create_tutorial,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
    initialize_collection,
    require_collection_root,
)
from ankiops.console import clickable_path, connect_or_exit, print_error
from ankiops.deck_sources import (
    RESERVED_MARKDOWN_FILES,
    DeckSource,
    discover_deck_sources,
    load_note_types_for_collection,
)
from ankiops.git import GitRepository, git_snapshot
from ankiops.image_widths import fix_image_widths_collection
from ankiops.interchange import (
    apply_deserialization_plan,
    plan_deserialize_from_file,
    serialize_to_file,
)
from ankiops.llm.commands import run_llm as run_llm_impl
from ankiops.llm.execution import run_task
from ankiops.llm.jobs import list_jobs as list_llm_jobs
from ankiops.llm.jobs import show_job
from ankiops.llm.planning import plan_task
from ankiops.llm.tasks import load_llm_task_catalog
from ankiops.media import (
    format_media_status,
    preflight_media_references,
    sync_all_media_from_anki,
    sync_all_media_to_anki,
)
from ankiops.note_types import sync_note_type_configs
from ankiops.note_types_command import run as run_note_type_impl
from ankiops.sync.from_anki import sync_collection_from_anki
from ankiops.sync.report import CollectionReport
from ankiops.sync.state import SyncState
from ankiops.sync.to_anki import sync_collection_to_anki

logger = logging.getLogger(__name__)

sync_media_from_anki = sync_all_media_from_anki
sync_media_to_anki = sync_all_media_to_anki


def _sync_note_types(anki, collection_root, state):
    return sync_note_type_configs(
        anki,
        load_note_types_for_collection(collection_root),
        sync_state=state,
    )


def _checkpoint_collab_repositories(collection_root: Path, action: str) -> None:
    for source in discover_deck_sources(collection_root)[1:]:
        commit = GitRepository(source.root).checkpoint(
            f"AnkiOps: snapshot before {action}"
        )
        if commit:
            logger.info(
                "Auto-committed %s checkpoint %s", source.display_name, commit[:7]
            )


def _local_markdown_paths(collection_root: Path) -> list[Path]:
    paths = DeckSource.local(collection_root).deck_files()
    paths.extend(_deleted_local_markdown_paths(collection_root))
    return list(dict.fromkeys(paths))


def _deleted_local_markdown_paths(collection_root: Path) -> list[Path]:
    try:
        result = GitRepository(collection_root).run(
            ["ls-files", "-z", "--deleted"],
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    paths = []
    for rel_path in result.stdout.split("\0"):
        if not rel_path:
            continue
        path = collection_root / rel_path
        if _is_local_markdown_deck_path(collection_root, path):
            paths.append(path)
    return paths


def _is_local_markdown_deck_path(collection_root: Path, path: Path) -> bool:
    return (
        path.parent == collection_root
        and path.suffix == ".md"
        and path.name.upper() not in RESERVED_MARKDOWN_FILES
        and "___" not in path.stem
    )


def run_init(args):
    """Initialize the current directory as an AnkiOps collection."""
    anki = connect_or_exit()
    profile = anki.get_active_profile()

    collection_root = initialize_collection(profile)

    if args.tutorial:
        create_tutorial(collection_root)

    logger.info(
        f"Initialized AnkiOps collection in {collection_root} (profile: {profile}). "
    )
    logger.info(
        f"If the AnkiOps add-on is installed, set ankiops_dir to: {collection_root}"
    )


def run_af(args):
    """Anki -> files: sync Anki changes into local files."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_root = require_collection_root(active_profile)
    logger.debug(f"Collection directory: {collection_root}")
    validate_collection_collab_sources(collection_root)
    preflight_media_references(collection_root)

    if not args.no_auto_commit:
        logger.debug("Creating pre-anki-to-files git snapshot")
        git_snapshot(
            collection_root,
            action="anki-to-files",
            paths=[
                *_local_markdown_paths(collection_root),
                collection_root / LOCAL_MEDIA_DIR,
            ],
        )
        _checkpoint_collab_repositories(collection_root, "anki-to-files")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")
    state = SyncState.open(collection_root)
    try:
        _run_af_with_state(anki, state, collection_root)
    finally:
        state.close()


def _run_af_with_state(anki, state: SyncState, collection_root: Path) -> None:
    logger.debug("Starting note sync (Anki -> files)")
    export_summary: CollectionReport = sync_collection_from_anki(
        anki=anki,
        state=state,
        collection_root=collection_root,
    )
    note_summary = export_summary.summary
    deck_count = len(export_summary.results)
    note_count = note_summary.total
    changes = note_summary.format()

    logger.info(
        f"Anki -> files: {deck_count} decks with {note_count} notes — {changes}"
    )
    for res in export_summary.results:
        deck_summary = res.summary
        deck_fmt = deck_summary.format()
        if deck_fmt != "no changes" and res.file_path:
            logger.info(
                f"  {clickable_path(res.file_path)}  {deck_fmt}",
                extra={"markup": True},
            )

    protected = export_summary.protected_note_groups
    if protected:
        protected_total = sum(group.note_count for group in protected)
        logger.warning(
            "Protected "
            f"{protected_total} local markdown note(s) without note_key comments "
            "during Anki -> files sync."
        )
        logger.warning(f"Affected deck file(s): {len(protected)}")
        for group in protected:
            logger.warning(f"  - {group.deck_name} ({group.note_count} notes)")
        logger.warning(
            "These notes were kept and not deleted. Use 'ankiops fa' to "
            "import them and assign note_key comments."
        )

    # Sync referenced media from Anki to local
    try:
        logger.debug("Starting media pull (Anki -> local)")
        media_result = sync_media_from_anki(anki, collection_root, state)
        logger.info(format_media_status(media_result, from_anki=True))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")


def _log_files_to_anki_errors(import_summary: CollectionReport) -> None:
    has_errors = False
    for result in import_summary.results:
        if not result.errors:
            continue

        if not has_errors:
            logger.error("Files -> Anki errors:")
            has_errors = True

        source = rich_escape(result.name or "unknown deck")
        if result.file_path:
            source = f"{source} ({clickable_path(result.file_path)})"

        for error in result.errors:
            logger.error(
                f"  {source}: {rich_escape(str(error))}",
                extra={"markup": True},
            )


def run_fa(args):
    """Files -> Anki: sync local file changes into Anki."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_root = require_collection_root(active_profile)
    logger.debug(f"Collection directory: {collection_root}")
    validate_collection_collab_sources(collection_root)
    preflight_media_references(collection_root)

    if not args.no_auto_commit:
        logger.debug("Creating pre-files-to-anki git snapshot")
        git_snapshot(
            collection_root,
            action="files-to-anki",
            paths=[
                *_local_markdown_paths(collection_root),
                collection_root / LOCAL_MEDIA_DIR,
                collection_root / NOTE_TYPES_DIR,
            ],
        )
        _checkpoint_collab_repositories(collection_root, "files-to-anki")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    state = SyncState.open(collection_root)
    try:
        _run_fa_with_state(anki, state, collection_root)
    finally:
        state.close()


def _run_fa_with_state(anki, state: SyncState, collection_root: Path) -> None:
    try:
        logger.debug("Starting media push (local -> Anki)")
        media_result = sync_media_to_anki(anki, collection_root, state)
        logger.info(format_media_status(media_result, from_anki=False))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")

    logger.debug("Starting note type sync")
    nt_summary = _sync_note_types(anki, collection_root, state)
    if nt_summary:
        logger.info(f"Note types: {nt_summary}")

    logger.debug("Starting note sync (files -> Anki)")
    import_summary: CollectionReport = sync_collection_to_anki(
        anki=anki,
        state=state,
        collection_root=collection_root,
    )
    note_summary = import_summary.summary
    deck_count = len(import_summary.results)
    note_count = note_summary.total
    untracked = import_summary.untracked_decks
    changes = note_summary.format()

    logger.info(
        f"Files -> Anki: {deck_count} decks with {note_count} notes — {changes}"
    )
    for res in import_summary.results:
        deck_summary = res.summary
        deck_fmt = deck_summary.format()
        if deck_fmt != "no changes" and res.file_path:
            logger.info(f"  {res.name}  {deck_fmt}")

    if note_summary.errors:
        _log_files_to_anki_errors(import_summary)

    protected = import_summary.protected_note_groups
    if protected:
        protected_total = sum(group.note_count for group in protected)
        logger.warning(
            f"Protected {protected_total} keyless Anki note(s) during "
            "files -> Anki sync."
        )
        logger.warning(f"Affected deck(s): {len(protected)}")
        for group in protected:
            logger.warning(f"  - {group.deck_name} ({group.note_count} notes)")
        logger.warning(
            "These notes were kept and not deleted because they do not have "
            "an AnkiOps Key. Add note_key comments in markdown and run "
            "'ankiops fa' to bring them under sync management."
        )

    if untracked:
        logger.warning(
            f"Found {len(untracked)} untracked Anki deck(s) with AnkiOps notes "
            "that are not present in your local markdown collection."
        )
        for deck in untracked:
            logger.warning(f"  - {deck.deck_name} ({len(deck.note_ids)} notes)")
        logger.warning("Use 'ankiops af' to bring them into your files.")

    if note_summary.errors:
        logger.critical(
            "Review and resolve errors above, then re-run files -> Anki — "
            "or you risk losing notes with the next Anki -> files sync."
        )


def run_serialize(args):
    """Serialize collection to JSON format."""
    if args.no_subdecks and not args.deck:
        logger.error("--no-subdecks requires --deck")
        raise SystemExit(2)

    collection_root = require_collection_root()

    if args.output:
        output_file = Path(args.output)
    elif args.deck:
        output_file = Path(f"{deck_name_to_file_stem(args.deck)}.json")
    else:
        output_file = Path("AnkiCollection.json")

    logger.debug(f"Serializing collection from: {collection_root}")
    logger.debug(f"Output file: {output_file}")

    try:
        serialize_to_file(
            collection_root,
            output_file,
            deck=args.deck,
            no_subdecks=args.no_subdecks,
        )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error


def run_deserialize(args):
    """Deserialize collection from JSON/ZIP format to target directory."""
    if args.input:
        serialized_file = Path(args.input)
    else:
        serialized_file = Path("AnkiCollection.json")

    if not serialized_file.exists():
        logger.error(f"Serialized file not found: {serialized_file}")
        raise SystemExit(1)

    collection_root = require_collection_root()
    deserialize_plan = plan_deserialize_from_file(
        serialized_file,
        collection_root=collection_root,
    )
    if not args.no_auto_commit:
        logger.debug("Creating pre-deserialize git snapshot")
        git_snapshot(
            collection_root,
            action="deserializing",
            paths=list(deserialize_plan.target_paths),
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    apply_deserialization_plan(
        deserialize_plan,
        overwrite=args.overwrite,
        collection_root=collection_root,
    )


def run_fix_image_widths(args):
    """Fix Markdown image width annotations in local deck files."""
    if args.no_subdecks and not args.deck:
        logger.error("--no-subdecks requires --deck")
        raise SystemExit(2)

    collection_root = require_collection_root()
    selected_paths = _local_markdown_paths(collection_root)
    if args.deck:
        deck_filter = args.deck.strip()
        subdeck_scope = f"{deck_filter}::"
        selected_paths = [
            md_file
            for md_file in selected_paths
            if file_stem_to_deck_name(md_file.stem) == deck_filter
            or (
                not args.no_subdecks
                and file_stem_to_deck_name(md_file.stem).startswith(subdeck_scope)
            )
        ]

    if not args.no_auto_commit:
        logger.debug("Creating pre-image-width-fix git snapshot")
        git_snapshot(
            collection_root,
            action="fixing image widths",
            paths=selected_paths,
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    try:
        result = fix_image_widths_collection(
            collection_root,
            deck=args.deck,
            no_subdecks=args.no_subdecks,
            tolerance=args.tolerance,
            width=args.width,
        )
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error

    mode = (
        f"forced width {args.width}px"
        if args.width is not None
        else f"auto tolerance ±{args.tolerance}px"
    )
    logger.info(
        "Image widths: "
        f"{result.decks_checked} decks, {result.notes_checked} notes, "
        f"{result.images_checked} images checked ({mode}) — "
        f"{result.decks_changed} decks, {result.notes_changed} notes, "
        f"{result.images_changed} images changed"
    )
    if result.changed:
        logger.info("Only Markdown files were edited. Run 'ankiops fa' to sync.")


def run_llm(args):
    """Delegate LLM command handling to the LLM CLI module."""
    run_llm_impl(
        args,
        require_collection_root_fn=require_collection_root,
        load_note_type_configs_fn=(
            lambda note_types_dir: load_note_types_for_collection(note_types_dir.parent)
        ),
        load_llm_task_catalog_fn=load_llm_task_catalog,
        plan_task_fn=plan_task,
        run_task_fn=run_task,
        list_jobs_fn=list_llm_jobs,
        show_job_fn=show_job,
    )


def run_note_type(args):
    run_note_type_impl(args)


def run_collab(args):
    try:
        run_collab_impl(args)
    except ValueError as error:
        print_error(str(error))
        logger.debug("Collab command failed", exc_info=True)
        raise SystemExit(1) from error
    except subprocess.CalledProcessError as error:
        output = ((error.stdout or "") + (error.stderr or "")).strip()
        print_error(output or f"Git command failed with exit {error.returncode}.")
        logger.debug("Collab Git command failed", exc_info=True)
        raise SystemExit(1) from error

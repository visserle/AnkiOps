import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from rich.markup import escape as rich_escape

from ankiops.collection import (
    LOCAL_MEDIA_DIR,
    NOTE_TYPES_DIR,
    create_tutorial,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
    initialize_collection,
    require_collection_dir,
)
from ankiops.console import clickable_path, connect_or_exit
from ankiops.deck_sources import (
    DeckSource,
    RESERVED_MARKDOWN_FILES,
    discover_deck_sources,
    load_note_types_for_sources,
)
from ankiops.git import CollectionGit, git_snapshot
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
    sync_all_media_from_anki,
    sync_all_media_to_anki,
)
from ankiops.note_types import sync_note_type_configs
from ankiops.note_types_command import run as run_note_type_impl
from ankiops.shared import run as run_shared_impl
from ankiops.sync.from_anki import sync_collection_from_anki
from ankiops.sync.report import CollectionReport
from ankiops.sync.state import SyncState
from ankiops.sync.to_anki import sync_collection_to_anki

logger = logging.getLogger(__name__)

sync_media_from_anki = sync_all_media_from_anki
sync_media_to_anki = sync_all_media_to_anki


def sync_note_types(anki_port, collection_dir, note_types_dir, db_port):
    sources = discover_deck_sources(collection_dir, note_types_dir=note_types_dir)
    configs = [
        config
        for source_config in load_note_types_for_sources(sources)
        for config in source_config.note_types
    ]
    return sync_note_type_configs(anki_port, configs, sync_state=db_port)


def _has_shared_sources(collection_dir: Path) -> bool:
    sources = discover_deck_sources(
        collection_dir,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )
    return any(source.is_shared for source in sources)


def _snapshot_paths(
    collection_dir: Path,
    scoped_paths: Sequence[Path],
    *,
    has_shared_scope: bool = False,
) -> list[Path]:
    if has_shared_scope or _has_shared_sources(collection_dir):
        return [collection_dir]
    return list(scoped_paths)


def _local_markdown_paths(collection_dir: Path) -> list[Path]:
    paths = DeckSource.local(collection_dir).deck_files()
    paths.extend(_deleted_local_markdown_paths(collection_dir))
    return _unique_paths(paths)


def _deleted_local_markdown_paths(collection_dir: Path) -> list[Path]:
    try:
        result = CollectionGit(collection_dir).run(
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
        path = collection_dir / rel_path
        if _is_local_markdown_deck_path(collection_dir, path):
            paths.append(path)
    return paths


def _is_local_markdown_deck_path(collection_dir: Path, path: Path) -> bool:
    return (
        path.parent == collection_dir
        and path.suffix == ".md"
        and path.name.upper() not in RESERVED_MARKDOWN_FILES
        and "___" not in path.stem
    )


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def run_init(args):
    """Initialize the current directory as an AnkiOps collection."""
    anki = connect_or_exit()
    profile = anki.get_active_profile()

    collection_dir = initialize_collection(profile)

    if args.tutorial:
        create_tutorial(collection_dir)

    logger.info(
        f"Initialized AnkiOps collection in {collection_dir} (profile: {profile}). "
        f"For the Anki add-on, set ankiops_dir to: {collection_dir}"
    )


def run_af(args):
    """Anki -> files: sync Anki changes into local files."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if not args.no_auto_commit:
        logger.debug("Creating pre-anki-to-files git snapshot")
        git_snapshot(
            collection_dir,
            action="anki-to-files",
            paths=_snapshot_paths(
                collection_dir,
                [
                    *_local_markdown_paths(collection_dir),
                    collection_dir / LOCAL_MEDIA_DIR,
                ],
            ),
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")
    db = SyncState.open(collection_dir)
    note_types_dir = collection_dir / NOTE_TYPES_DIR

    logger.debug("Starting note sync (Anki -> files)")
    export_summary: CollectionReport = sync_collection_from_anki(
        anki_port=anki,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
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
        media_result = sync_media_from_anki(anki, collection_dir, db)
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

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if not args.no_auto_commit:
        logger.debug("Creating pre-files-to-anki git snapshot")
        git_snapshot(
            collection_dir,
            action="files-to-anki",
            paths=_snapshot_paths(
                collection_dir,
                [
                    *_local_markdown_paths(collection_dir),
                    collection_dir / LOCAL_MEDIA_DIR,
                    collection_dir / NOTE_TYPES_DIR,
                ],
            ),
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    db = SyncState.open(collection_dir)
    note_types_dir = collection_dir / NOTE_TYPES_DIR

    try:
        logger.debug("Starting media push (local -> Anki)")
        media_result = sync_media_to_anki(anki, collection_dir, db)
        logger.info(format_media_status(media_result, from_anki=False))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")

    logger.debug("Starting note type sync")
    nt_summary = sync_note_types(anki, collection_dir, note_types_dir, db)
    if nt_summary:
        logger.info(f"Note types: {nt_summary}")

    logger.debug("Starting note sync (files -> Anki)")
    import_summary: CollectionReport = sync_collection_to_anki(
        anki_port=anki,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
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

    collection_dir = require_collection_dir()

    if args.output:
        output_file = Path(args.output)
    elif args.deck:
        output_file = Path(f"{deck_name_to_file_stem(args.deck)}.json")
    else:
        output_file = Path("AnkiCollection.json")

    logger.debug(f"Serializing collection from: {collection_dir}")
    logger.debug(f"Output file: {output_file}")

    try:
        serialize_to_file(
            collection_dir,
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

    collection_dir = require_collection_dir()
    deserialize_plan = plan_deserialize_from_file(
        serialized_file,
        collection_dir=collection_dir,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )
    if not args.no_auto_commit:
        logger.debug("Creating pre-deserialize git snapshot")
        git_snapshot(
            collection_dir,
            action="deserializing",
            paths=_snapshot_paths(
                collection_dir,
                deserialize_plan.target_paths,
                has_shared_scope=deserialize_plan.has_shared_sources,
            ),
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    apply_deserialization_plan(
        deserialize_plan,
        overwrite=args.overwrite,
        collection_dir=collection_dir,
    )


def run_fix_image_widths(args):
    """Fix Markdown image width annotations in local deck files."""
    if args.no_subdecks and not args.deck:
        logger.error("--no-subdecks requires --deck")
        raise SystemExit(2)

    collection_dir = require_collection_dir()
    selected_paths = _local_markdown_paths(collection_dir)
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
            collection_dir,
            action="fixing image widths",
            paths=_snapshot_paths(collection_dir, selected_paths),
        )
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    try:
        result = fix_image_widths_collection(
            collection_dir,
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
    """Delegates LLM command handling to the LLM CLI module."""
    run_llm_impl(
        args,
        require_collection_dir_fn=require_collection_dir,
        load_note_type_configs_fn=(
            lambda note_types_dir: [
                config
                for source_config in load_note_types_for_sources(
                    discover_deck_sources(
                        note_types_dir.parent,
                        note_types_dir=note_types_dir,
                    )
                )
                for config in source_config.note_types
            ]
        ),
        load_llm_task_catalog_fn=load_llm_task_catalog,
        plan_task_fn=plan_task,
        run_task_fn=run_task,
        list_jobs_fn=list_llm_jobs,
        show_job_fn=show_job,
    )


def run_note_type(args):
    run_note_type_impl(args)


def run_shared(args):
    try:
        if getattr(args, "shared_command", None) == "submit" and args.from_anki:
            run_af(SimpleNamespace(no_auto_commit=False))
        run_shared_impl(args)
        if getattr(args, "shared_command", None) == "update" and args.to_anki:
            run_fa(SimpleNamespace(no_auto_commit=False))
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error
    except subprocess.CalledProcessError as error:
        output = ((error.stdout or "") + (error.stderr or "")).strip()
        logger.error(output or f"git command failed with exit {error.returncode}")
        raise SystemExit(1) from error

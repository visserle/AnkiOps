import argparse
import logging
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

from rich.markup import escape as rich_escape

from ankiops.anki_client import AnkiConnectionError
from ankiops.cli_anki import connect_or_exit
from ankiops.collab import run as run_collab_impl
from ankiops.config import (
    NOTE_TYPES_DIR,
    deck_name_to_file_stem,
    require_collection_dir,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.image_widths import fix_image_widths_collection
from ankiops.import_notes import import_collection
from ankiops.init import create_tutorial, initialize_collection
from ankiops.llm.commands import configure_llm_parser
from ankiops.llm.commands import run_llm as run_llm_impl
from ankiops.llm.config_loader import load_llm_task_catalog
from ankiops.llm.runner import list_jobs as list_llm_jobs
from ankiops.llm.runner import plan_task, run_task, show_job
from ankiops.log import clickable_path, configure_logging
from ankiops.models import CollectionResult
from ankiops.note_type_cli import run as run_note_type
from ankiops.serializer import (
    deserialize_from_file,
    serialize_to_file,
)
from ankiops.sources import discover_sync_sources, load_configs_for_sources
from ankiops.sync_media import (
    format_media_status,
    sync_all_media_from_anki,
    sync_all_media_to_anki,
)
from ankiops.sync_note_types import sync_note_type_configs

logger = logging.getLogger(__name__)

sync_media_from_anki = sync_all_media_from_anki
sync_media_to_anki = sync_all_media_to_anki


def sync_note_types(anki_port, fs_port, collection_dir, note_types_dir, db_port):
    sources = discover_sync_sources(collection_dir, note_types_dir=note_types_dir)
    configs = [
        config
        for source_config in load_configs_for_sources(sources)
        for config in source_config.configs
    ]
    return sync_note_type_configs(anki_port, configs, db_port=db_port)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


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


def run_am(args):
    """Anki -> Markdown: export decks to markdown files."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if not args.no_auto_commit:
        logger.debug("Creating pre-export git snapshot")
        git_snapshot(collection_dir, "export")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")
    fs = FileSystemAdapter()
    db = SQLiteDbAdapter.open(collection_dir)
    note_types_dir = collection_dir / NOTE_TYPES_DIR

    logger.debug("Starting note export (Anki -> Markdown)")
    export_summary: CollectionResult = export_collection(
        anki_port=anki,
        fs_port=fs,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
    )
    note_summary = export_summary.summary
    deck_count = len(export_summary.results)
    note_count = note_summary.total
    changes = note_summary.format()

    logger.info(f"Export: {deck_count} decks with {note_count} notes — {changes}")
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
            "during export."
        )
        logger.warning(f"Affected deck file(s): {len(protected)}")
        for group in protected:
            logger.warning(f"  - {group.deck_name} ({group.note_count} notes)")
        logger.warning(
            "These notes were kept and not deleted. Use 'ankiops ma' to "
            "import them and assign note_key comments."
        )

    # Sync referenced media from Anki to local
    try:
        logger.debug("Starting media pull (Anki -> local)")
        media_result = sync_media_from_anki(anki, fs, collection_dir, db)
        logger.info(format_media_status(media_result, from_anki=True))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")


def _log_import_errors(import_summary: CollectionResult) -> None:
    has_errors = False
    for result in import_summary.results:
        if not result.errors:
            continue

        if not has_errors:
            logger.error("Import errors:")
            has_errors = True

        source = rich_escape(result.name or "unknown deck")
        if result.file_path:
            source = f"{source} ({clickable_path(result.file_path)})"

        for error in result.errors:
            logger.error(
                f"  {source}: {rich_escape(str(error))}",
                extra={"markup": True},
            )


def run_ma(args):
    """Markdown -> Anki: import markdown files into Anki."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if not args.no_auto_commit:
        logger.debug("Creating pre-import git snapshot")
        git_snapshot(collection_dir, "import")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    fs = FileSystemAdapter()
    db = SQLiteDbAdapter.open(collection_dir)
    note_types_dir = collection_dir / NOTE_TYPES_DIR

    try:
        logger.debug("Starting media push (local -> Anki)")
        media_result = sync_media_to_anki(anki, fs, collection_dir, db)
        logger.info(format_media_status(media_result, from_anki=False))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")

    logger.debug("Starting note type sync")
    nt_summary = sync_note_types(anki, fs, collection_dir, note_types_dir, db)
    if nt_summary:
        logger.info(f"Note types: {nt_summary}")

    logger.debug("Starting note import (Markdown -> Anki)")
    import_summary: CollectionResult = import_collection(
        anki_port=anki,
        fs_port=fs,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
    )
    note_summary = import_summary.summary
    deck_count = len(import_summary.results)
    note_count = note_summary.total
    untracked = import_summary.untracked_decks
    changes = note_summary.format()

    logger.info(f"Import: {deck_count} decks with {note_count} notes — {changes}")
    for res in import_summary.results:
        deck_summary = res.summary
        deck_fmt = deck_summary.format()
        if deck_fmt != "no changes" and res.file_path:
            logger.info(f"  {res.name}  {deck_fmt}")

    if note_summary.errors:
        _log_import_errors(import_summary)

    protected = import_summary.protected_note_groups
    if protected:
        protected_total = sum(group.note_count for group in protected)
        logger.warning(
            f"Protected {protected_total} keyless Anki note(s) during import."
        )
        logger.warning(f"Affected deck(s): {len(protected)}")
        for group in protected:
            logger.warning(f"  - {group.deck_name} ({group.note_count} notes)")
        logger.warning(
            "These notes were kept and not deleted because they do not have "
            "an AnkiOps Key. Add note_key comments in markdown and re-import "
            "to bring them under sync management."
        )

    if untracked:
        logger.warning(
            f"Found {len(untracked)} untracked Anki deck(s) with AnkiOps notes "
            "that are not present in your local markdown collection."
        )
        for deck in untracked:
            logger.warning(f"  - {deck.deck_name} ({len(deck.note_ids)} notes)")
        logger.warning("Use 'ankiops am' to bring them into your collection.")

    if note_summary.errors:
        logger.critical(
            "Review and resolve errors above, then re-run the import — "
            "or you risk losing notes with the next export."
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
    if not args.no_auto_commit:
        logger.debug("Creating pre-deserialize git snapshot")
        git_snapshot(collection_dir, "deserialize")
    else:
        logger.debug("Auto-commit disabled (--no-auto-commit)")

    deserialize_from_file(
        serialized_file,
        overwrite=args.overwrite,
        collection_dir=collection_dir,
    )


def run_fix_image_widths(args):
    """Fix Markdown image width annotations in local deck files."""
    if args.no_subdecks and not args.deck:
        logger.error("--no-subdecks requires --deck")
        raise SystemExit(2)

    collection_dir = require_collection_dir()

    if not args.no_auto_commit:
        logger.debug("Creating pre-image-width-fix git snapshot")
        git_snapshot(collection_dir, "fix-image-widths")
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
        logger.info("Only Markdown files were edited. Run 'ankiops ma' to sync.")


def run_llm(args):
    """Delegates LLM command handling to the LLM CLI module."""
    run_llm_impl(
        args,
        require_collection_dir_fn=require_collection_dir,
        load_note_type_configs_fn=(
            lambda note_types_dir: [
                config
                for source_config in load_configs_for_sources(
                    discover_sync_sources(
                        note_types_dir.parent,
                        note_types_dir=note_types_dir,
                    )
                )
                for config in source_config.configs
            ]
        ),
        load_llm_task_catalog_fn=load_llm_task_catalog,
        plan_task_fn=plan_task,
        run_task_fn=run_task,
        list_jobs_fn=list_llm_jobs,
        show_job_fn=show_job,
    )


def run_collab(args):
    try:
        if getattr(args, "collab_command", None) == "contribute" and args.from_anki:
            run_am(SimpleNamespace(no_auto_commit=False))
        run_collab_impl(args)
        if getattr(args, "collab_command", None) == "pull" and args.to_anki:
            run_ma(SimpleNamespace(no_auto_commit=False))
    except ValueError as error:
        logger.error(str(error))
        raise SystemExit(1) from error
    except subprocess.CalledProcessError as error:
        output = ((error.stdout or "") + (error.stderr or "")).strip()
        logger.error(output or f"git command failed with exit {error.returncode}")
        raise SystemExit(1) from error


def _get_cli_version() -> str:
    try:
        return version("ankiops")
    except PackageNotFoundError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="AnkiOps – A bridge between Anki and your filesystem..",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_cli_version()}",
        help="Show version and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Init parser
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize current directory as an AnkiOps collection",
    )
    init_parser.add_argument(
        "--tutorial",
        action="store_true",
        help="Create a tutorial markdown file in the collection directory",
    )
    init_parser.set_defaults(handler=run_init)

    # Anki to Markdown (am) parser
    am_parser = subparsers.add_parser(
        "anki-to-markdown",
        aliases=["am"],
        help="Anki -> Markdown (export)",
    )
    am_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    am_parser.set_defaults(handler=run_am)

    # Markdown to Anki (ma) parser
    ma_parser = subparsers.add_parser(
        "markdown-to-anki",
        aliases=["ma"],
        help="Markdown -> Anki (import)",
    )
    ma_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    ma_parser.set_defaults(handler=run_ma)

    # Serialize parser
    serialize_parser = subparsers.add_parser(
        "serialize",
        help="Export collection to portable JSON format",
    )
    serialize_parser.add_argument(
        "--output",
        "-o",
        help=(
            "Output file path (default: AnkiCollection.json, "
            "<deck-stem>.json when --deck is set)"
        ),
    )
    serialize_parser.add_argument(
        "--deck",
        help="Serialize only this deck (includes subdecks by default)",
    )
    serialize_parser.add_argument(
        "--no-subdecks",
        action="store_true",
        help="With --deck, serialize only the exact deck (exclude subdecks)",
    )
    serialize_parser.set_defaults(handler=run_serialize)

    # Deserialize parser
    deserialize_parser = subparsers.add_parser(
        "deserialize",
        help="Import markdown from JSON into an initialized collection",
    )
    deserialize_parser.add_argument(
        "--input",
        "-i",
        help="Input file path (default: <collection-name>.json)",
    )
    deserialize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files",
    )
    deserialize_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    deserialize_parser.set_defaults(handler=run_deserialize)

    # Image width fixer parser
    fix_widths_parser = subparsers.add_parser(
        "fix-image-widths",
        help="Normalize or force Markdown image widths via local file edits",
    )
    fix_widths_parser.add_argument(
        "--deck",
        help="Fix only this deck (includes subdecks by default)",
    )
    fix_widths_parser.add_argument(
        "--no-subdecks",
        action="store_true",
        help="With --deck, fix only the exact deck (exclude subdecks)",
    )
    fix_widths_parser.add_argument(
        "--tolerance",
        type=_non_negative_int,
        default=5,
        help="Pixel tolerance for auto mode (default: 5)",
    )
    fix_widths_parser.add_argument(
        "--width",
        type=_positive_int,
        help="Force every Markdown image in scope to this width in pixels",
    )
    fix_widths_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    fix_widths_parser.set_defaults(handler=run_fix_image_widths)

    configure_llm_parser(subparsers, handler=run_llm)

    collab_parser = subparsers.add_parser(
        "collab",
        help="Publish, subscribe to, update, and contribute GitHub collab decks",
    )
    collab_subparsers = collab_parser.add_subparsers(
        dest="collab_command",
        required=True,
    )

    collab_publish = collab_subparsers.add_parser(
        "publish",
        help="Publish a local deck tree to a GitHub collab source",
    )
    collab_publish.add_argument("deck", help="Deck to publish (includes subdecks)")
    collab_publish.add_argument(
        "repo",
        help="GitHub repo as owner/repo (letters, digits, hyphens)",
    )
    publish_visibility = collab_publish.add_mutually_exclusive_group()
    publish_visibility.add_argument(
        "--public",
        action="store_true",
        help="Create the GitHub repo as public when it does not exist",
    )
    publish_visibility.add_argument(
        "--private",
        action="store_false",
        dest="public",
        help="Create the GitHub repo as private when it does not exist (default)",
    )
    collab_publish.set_defaults(public=False)
    collab_publish.set_defaults(handler=run_collab)

    collab_subscribe = collab_subparsers.add_parser(
        "subscribe",
        help="Add a GitHub collab source to this collection",
    )
    collab_subscribe.add_argument(
        "repo",
        help="GitHub repo as owner/repo (letters, digits, hyphens)",
    )
    collab_subscribe.set_defaults(handler=run_collab)

    collab_pull = collab_subparsers.add_parser(
        "pull",
        help="Pull one or all collab sources from GitHub",
    )
    collab_pull.add_argument(
        "repo",
        nargs="?",
        help="GitHub repo as owner/repo (letters, digits, hyphens)",
    )
    collab_pull.add_argument(
        "--to-anki",
        action="store_true",
        help="Run Markdown -> Anki after pulling files",
    )
    collab_pull.set_defaults(handler=run_collab)

    collab_contribute = collab_subparsers.add_parser(
        "contribute",
        help="Prepare a GitHub PR from local collab source edits",
    )
    collab_contribute.add_argument(
        "repo",
        help="GitHub repo as owner/repo (letters, digits, hyphens)",
    )
    collab_contribute.add_argument(
        "--from-anki",
        action="store_true",
        help="Run Anki -> Markdown before preparing the contribution",
    )
    collab_contribute.set_defaults(handler=run_collab)

    collab_status = collab_subparsers.add_parser(
        "status",
        help="Show known collab sources",
    )
    collab_status.set_defaults(handler=run_collab)

    note_types_parser = subparsers.add_parser(
        "note-types",
        help="Show note type overview or add a note type from Anki",
        description=(
            "Show note type overview by default. "
            "Use '--add <name>' to copy a note type from Anki."
        ),
    )
    note_types_parser.set_defaults(handler=run_note_type, action="list")
    note_types_parser.add_argument(
        "--add",
        dest="add_name",
        metavar="NAME",
        help="Note type name to copy from Anki",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(stream_level=log_level)

    if hasattr(args, "handler"):
        try:
            args.handler(args)
        except AnkiConnectionError as error:
            logger.error(
                "Error communicating with Anki. Make sure Anki is running and "
                "either AnkiOpsConnect or AnkiConnect is enabled."
            )
            logger.debug(f"Anki connection error details: {error}")
            raise SystemExit(1) from error
    else:
        # Show welcome screen when no command is provided
        cli_version = _get_cli_version()
        print("=" * 60)
        print(f"AnkiOps v{cli_version} – A bidirectional Anki-Markdown bridge")
        print("=" * 60)
        print()
        print("Available commands:")
        print(
            "  init              Initialize current directory as an AnkiOps collection"
        )
        print("  anki-to-markdown  Export Anki decks to Markdown files (alias: am)")
        print("  markdown-to-anki  Import Markdown files into Anki (alias: ma)")
        print("  serialize         Serialize Markdown decks to JSON format")
        print("  deserialize       Deserialize JSON file to Markdown decks")
        print("  fix-image-widths  Normalize or force Markdown image widths")
        print("  llm               Status/plan/run LLM jobs and inspect one LLM job")
        print("  note-types        List note type labels or add note types from Anki")
        print()
        print("Usage examples:")
        print(
            "  ankiops init --tutorial                      # Initialize with tutorial"
        )
        print(
            "  ankiops am                                   "
            "# Export all decks to Markdown"
        )
        print(
            "  ankiops ma                                   "
            "# Import all Markdown files to Anki"
        )
        print(
            "  ankiops serialize -o my-deck.json            "
            "# Serialize collection to file"
        )
        print(
            "  ankiops deserialize -i my-deck.json          "
            "# Deserialize file to markdown"
        )
        print(
            "  ankiops fix-image-widths                     "
            "# Normalize near-equal image widths"
        )
        print(
            "  ankiops llm                                  "
            "# Show LLM status dashboard (strict)"
        )
        print("  ankiops llm grammar                          # Dry-run plan")
        print("  ankiops llm grammar --run                    # Run task job")
        print(
            "  ankiops llm --job latest                     # Show most recent LLM job"
        )
        print("  ankiops llm --job <job_id>                   # Show one LLM job")
        print(
            "  ankiops note-types                           "
            "# Show note types and label registry"
        )
        print("  ankiops note-types --add MyNoteType     # Copy note type from Anki")
        print()
        print("For more information:")
        print("  ankiops --version              # Show installed version")
        print("  ankiops --help                 # Show general help")
        print("  ankiops <command> --help       # Show help for a specific command")
        print("=" * 60)


if __name__ == "__main__":
    main()

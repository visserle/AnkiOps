import argparse
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ankiops.cli_anki import connect_or_exit
from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    serialize_collection_to_json,
)
from ankiops.config import (
    ANKIOPS_DB,
    get_collection_dir,
    get_note_types_dir,
    require_collection_dir,
    require_initialized_collection_dir,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.import_notes import import_collection
from ankiops.init import create_tutorial, initialize_collection
from ankiops.llm.cli import configure_llm_parser
from ankiops.llm.cli import run_llm as run_llm_impl
from ankiops.llm.config_loader import load_llm_task_catalog
from ankiops.llm.runner import list_jobs as list_llm_jobs
from ankiops.llm.runner import plan_task, resume_task, run_task, show_job
from ankiops.log import clickable_path, configure_logging
from ankiops.models import CollectionResult
from ankiops.note_type_cli import run as run_note_type
from ankiops.sync_media import sync_media_from_anki, sync_media_to_anki
from ankiops.sync_note_types import sync_note_types

logger = logging.getLogger(__name__)


def _get_cli_version() -> str:
    try:
        return version("ankiops")
    except PackageNotFoundError:
        return "unknown"


def _format_media_status(media_result, *, from_anki: bool) -> str:
    checked = media_result.checked
    summary = media_result.summary

    if checked == 0:
        return "Media: no referenced files"

    if media_result.missing:
        if from_anki:
            return (
                f"Media: {checked} files checked — "
                f"{summary.synced} pulled, {media_result.missing} missing in Anki"
            )
        return (
            f"Media: {checked} files checked — "
            f"{summary.format()}, {media_result.missing} missing locally"
        )

    return f"Media: {checked} files checked — {summary.format()}"


def _log_import_errors(import_summary: CollectionResult) -> None:
    has_errors = False
    for result in import_summary.results:
        if not result.errors:
            continue

        if not has_errors:
            logger.error("Import errors:")
            has_errors = True

        source = result.name or "unknown deck"
        if result.file_path:
            source = f"{source} ({clickable_path(result.file_path)})"

        for error in result.errors:
            logger.error(f"  {source}: {error}")


def run_init(args):
    """Initialize the current directory as an AnkiOps collection."""
    anki = connect_or_exit()
    profile = anki.get_active_profile()

    collection_dir = initialize_collection(profile)

    if args.tutorial:
        create_tutorial(collection_dir)

    logger.info(
        f"Initialized AnkiOps collection in {collection_dir} (profile: {profile})"
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
    note_types_dir = get_note_types_dir()

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
            logger.info(f"  {clickable_path(res.file_path)}  {deck_fmt}")

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
        logger.info(_format_media_status(media_result, from_anki=True))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")


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
    note_types_dir = get_note_types_dir()

    try:
        logger.debug("Starting media push (local -> Anki)")
        media_result = sync_media_to_anki(anki, fs, collection_dir, db)
        logger.info(_format_media_status(media_result, from_anki=False))
    except Exception as error:
        logger.warning(f"Media sync failed: {error}")

    logger.debug("Starting note type sync")
    nt_summary = sync_note_types(anki, fs, note_types_dir, db)
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

    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"{collection_dir.name}.json")

    logger.debug(f"Serializing collection from: {collection_dir}")
    logger.debug(f"Output file: {output_file}")

    try:
        serialize_collection_to_json(
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
        collection_dir = get_collection_dir()
        serialized_file = Path(f"{collection_dir.name}.json")

    if not serialized_file.exists():
        logger.error(f"Serialized file not found: {serialized_file}")
        raise SystemExit(1)

    deserialize_collection_from_json(
        serialized_file,
        overwrite=args.overwrite,
    )


def run_llm(args):
    """Delegates LLM command handling to the LLM CLI module."""
    run_llm_impl(
        args,
        require_initialized_collection_dir_fn=require_initialized_collection_dir,
        get_note_types_dir_fn=get_note_types_dir,
        load_note_type_configs_fn=(
            lambda note_types_dir: FileSystemAdapter().load_note_type_configs(
                note_types_dir
            )
        ),
        load_llm_task_catalog_fn=load_llm_task_catalog,
        plan_task_fn=plan_task,
        run_task_fn=run_task,
        resume_task_fn=resume_task,
        list_jobs_fn=list_llm_jobs,
        show_job_fn=show_job,
    )


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
        help="Output file path (default: <collection-name>.json)",
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
        help="Import markdown from JSON (run 'init' after to set up)",
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
    deserialize_parser.set_defaults(handler=run_deserialize)

    configure_llm_parser(subparsers, handler=run_llm)

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
        args.handler(args)
    else:
        # Show welcome screen when no command is provided
        cli_version = _get_cli_version()
        print("=" * 60)
        print(f"AnkiOps v{cli_version} – A bridge between Anki and your filesystem.")
        print("=" * 60)
        print()
        print("Available commands:")
        print(
            "  init              Initialize current directory as an AnkiOps collection"
        )
        print("  anki-to-markdown  Export Anki decks to Markdown files (alias: am)")
        print("  markdown-to-anki  Import Markdown files into Anki (alias: ma)")
        print("  serialize         Export collection to a portable JSON/ZIP file")
        print("  deserialize       Import markdown/media from JSON/ZIP")
        print("  llm               Status/plan/run LLM jobs and inspect one LLM job")
        print(
            "  note-types        List note type labels or import note types from Anki"
        )
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

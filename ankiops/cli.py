import argparse
import logging
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    serialize_collection_to_json,
)
from ankiops.config import (
    ANKIOPS_DB,
    get_collection_dir,
    get_note_types_dir,
    require_collection_dir,
)
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.import_notes import import_collection
from ankiops.init import create_tutorial, initialize_collection
from ankiops.log import clickable_path, configure_logging
from ankiops.models import CollectionExportResult, CollectionImportResult
from ankiops.sync_media import sync_media_from_anki, sync_media_to_anki
from ankiops.sync_note_types import sync_note_types

logger = logging.getLogger(__name__)


def connect_or_exit() -> AnkiAdapter:
    """Verify AnkiConnect is reachable; exit on failure. Returns the adapter."""
    anki = AnkiAdapter()
    try:
        version = anki.get_version()
        logger.debug(f"Connected to AnkiConnect (version {version})")
    except Exception as e:
        logger.error(f"Error connecting to AnkiConnect: {e}")
        logger.error("Make sure Anki is running and AnkiConnect is installed.")
        raise SystemExit(1)
    return anki


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
        git_snapshot(collection_dir, "export")
    fs = FileSystemAdapter()
    db = SQLiteDbAdapter.load(collection_dir)
    note_types_dir = get_note_types_dir()

    export_summary: CollectionExportResult = export_collection(
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

    logger.info(
        f"Export: {deck_count} decks with {note_count} notes — {changes}"
    )
    for res in export_summary.results:
        s = res.summary
        deck_fmt = s.format()
        if deck_fmt != "no changes" and res.file_path:
            logger.info(f"  {clickable_path(res.file_path)}  {deck_fmt}")

    # Sync referenced media from Anki to local
    try:
        media_result = sync_media_from_anki(anki, fs, collection_dir, db)
        media_summary = media_result.summary
        if media_summary.format() != "no changes":
            logger.info(
                f"Media: {media_summary.total} files — {media_summary.format()}"
            )
        else:
            logger.debug("Media: no changes")
    except Exception as e:
        logger.warning(f"Media sync failed: {e}")


def run_ma(args):
    """Markdown -> Anki: import markdown files into Anki."""
    anki = connect_or_exit()
    active_profile = anki.get_active_profile()

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")
    fs = FileSystemAdapter()
    db = SQLiteDbAdapter.load(collection_dir)
    note_types_dir = get_note_types_dir()

    try:
        media_result = sync_media_to_anki(anki, fs, collection_dir, db)
        media_summary = media_result.summary
        if media_summary.format() != "no changes":
            logger.info(
                f"Media: {media_summary.total} files — {media_summary.format()}"
            )
        else:
            logger.debug("Media: no changes")
    except Exception as e:
        logger.warning(f"Media sync failed: {e}")

    nt_summary = sync_note_types(anki, fs, note_types_dir)
    if nt_summary:
        logger.info(f"Note types: {nt_summary}")

    if not args.no_auto_commit:
        git_snapshot(collection_dir, "import")

    import_summary: CollectionImportResult = import_collection(
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

    logger.info(
        f"Import: {deck_count} decks with {note_count} notes — {changes}"
    )
    for res in import_summary.results:
        s = res.summary
        deck_fmt = s.format()
        if deck_fmt != "no changes" and res.file_path:
            logger.info(f"  {res.deck_name}  {deck_fmt}")

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

    serialize_collection_to_json(
        collection_dir,
        output_file,
        no_ids=args.no_ids,
    )


def run_deserialize(args):
    """Deserialize collection from JSON/ZIP format to target directory."""
    serialized_file = Path(args.serialized_file)

    if not serialized_file.exists():
        logger.error(f"Serialized file not found: {serialized_file}")
        raise SystemExit(1)

    deserialize_collection_from_json(
        serialized_file, overwrite=args.overwrite, no_ids=args.no_ids
    )


def main():
    parser = argparse.ArgumentParser(
        description="AnkiOps – Manage Anki decks as Markdown files.",
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
    serialize_parser.set_defaults(handler=run_serialize)

    # Deserialize parser
    deserialize_parser = subparsers.add_parser(
        "deserialize",
        help="Import markdown from JSON (run 'init' after to set up)",
    )
    deserialize_parser.add_argument(
        "serialized_file",
        metavar="FILE",
        help="Serialized file to import (.json)",
    )
    deserialize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files",
    )
    deserialize_parser.set_defaults(handler=run_deserialize)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(stream_level=log_level, ignore_libs=["urllib3.connectionpool"])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        # Show welcome screen when no command is provided
        print("=" * 60)
        print("AnkiOps – Manage Anki decks as Markdown files")
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
        print()
        print("Usage examples:")
        print("  ankiops init --tutorial            # Initialize with tutorial")
        print("  ankiops am                         # Export all decks to Markdown")
        print(
            "  ankiops ma                         # Import all Markdown files to Anki"
        )
        print("  ankiops serialize -o my-deck.json  # Serialize collection to file")
        print("  ankiops deserialize my-deck.json   # Deserialize file, then run init")
        print()
        print("For more information:")
        print("  ankiops --help                 # Show general help")
        print("  ankiops <command> --help       # Show help for a specific command")
        print("=" * 60)


if __name__ == "__main__":
    main()

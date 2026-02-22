import argparse
import logging
from pathlib import Path

from ankiops.anki_client import invoke
from ankiops.anki_to_markdown import (
    export_collection,
    export_deck,
)
from ankiops.collection_serializer import (
    deserialize_collection_from_json,
    serialize_collection_to_json,
)
from ankiops.config import (
    ANKIOPS_DB,
    get_collection_dir,
    require_collection_dir,
)
from ankiops.git import git_snapshot
from ankiops.init import create_tutorial, initialize_collection
from ankiops.log import configure_logging
from ankiops.markdown_to_anki import (
    import_collection,
    import_file,
)
from ankiops.models import (
    CollectionExportResult,
    CollectionImportResult,
)
from ankiops.note_type_sync import ensure_note_types
from ankiops.media_sync import (
    extract_media_references,
    sync_from_anki,
    sync_to_anki,
)

logger = logging.getLogger(__name__)


def connect_or_exit():
    """Verify AnkiConnect is reachable; exit on failure."""
    try:
        version = invoke("version")
        logger.debug(f"Connected to AnkiConnect (version {version})")
    except Exception as e:
        logger.error(f"Error connecting to AnkiConnect: {e}")
        logger.error("Make sure Anki is running and AnkiConnect is installed.")
        raise SystemExit(1)


def run_init(args):
    """Initialize the current directory as an AnkiOps collection."""
    connect_or_exit()
    profile = invoke("getActiveProfile")

    collection_dir = initialize_collection(profile)

    if args.tutorial:
        create_tutorial(collection_dir)

    logger.info(
        f"Initialized AnkiOps collection in {collection_dir} (profile: {profile})"
    )


def run_am(args):
    """Anki -> Markdown: export decks to markdown files."""
    connect_or_exit()
    active_profile = invoke("getActiveProfile")

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if not args.no_auto_commit:
        git_snapshot(collection_dir, "export")

    if args.deck:
        logger.debug(f"Processing deck: {args.deck}...")
        result = export_deck(args.deck, output_dir=str(collection_dir))
        note_summary = result.summary
        files_count = 1 if result.file_path else 0
    else:
        export_summary: CollectionExportResult = export_collection(
            output_dir=str(collection_dir),
            keep_orphans=args.keep_orphans,
        )
        note_summary = export_summary.summary
        files_count = len(export_summary.results)

    logger.info(f"Export complete: {files_count} files — {note_summary}")

    # Sync referenced media from Anki to local
    try:
        media_dir = invoke("getMediaDirPath")
        all_refs = set()
        for md_file in collection_dir.glob("*.md"):
            all_refs.update(extract_media_references(md_file.read_text()))

        if all_refs:
            logger.debug("Syncing referenced media from Anki...")
            media_result = sync_from_anki(collection_dir, Path(media_dir), all_refs)
            media_summary = media_result.summary
            if media_summary.format() != "no changes":
                logger.info(
                    f"Media sync complete: {media_summary.total} files — {media_summary}"
                )
            else:
                logger.debug("Media sync complete: no changes")
    except Exception as e:
        logger.warning(f"Media sync failed: {e}")


def run_ma(args):
    """Markdown -> Anki: import markdown files into Anki."""
    connect_or_exit()
    active_profile = invoke("getActiveProfile")

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    # Sync local media to Anki (and rename if needed)
    try:
        media_dir = invoke("getMediaDirPath")
        media_result = sync_to_anki(collection_dir, Path(media_dir))
        media_summary = media_result.summary
        if media_summary.format() != "no changes":
            logger.info(
                f"Media sync complete: {media_summary.total} files — {media_summary}"
            )
        else:
            logger.debug("Media sync complete: no changes")
    except Exception as e:
        logger.warning(f"Media sync failed: {e}")

    ensure_note_types()

    if not args.no_auto_commit:
        git_snapshot(collection_dir, "import")

    deleted_notes = 0

    if args.file:
        logger.debug(f"Processing {Path(args.file).name}...")
        result = import_file(
            Path(args.file),
            only_add_new=args.only_add_new,
        )
        note_summary = result.summary
        files_stat = 1
        untracked = []
    else:
        import_summary: CollectionImportResult = import_collection(
            str(collection_dir), only_add_new=args.only_add_new
        )
        note_summary = import_summary.summary
        files_stat = len(import_summary.results)
        untracked = import_summary.untracked_decks

    # Add back explicit deleted notes from CLI prompt
    note_summary.deleted += deleted_notes

    logger.info(f"Import complete: {files_stat} files — {note_summary}")

    if untracked:
        logger.warning(
            f"Found {len(untracked)} untracked Anki deck(s) with AnkiOps notes "
            "that are not present in your local markdown collection."
        )
        for deck in untracked:
            logger.warning(f"  - {deck.deck_name} ({len(deck.note_ids)} notes)")
        logger.warning("Use 'ankiops export' to bring them into your collection.")

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
        "--deck",
        "-d",
        help="Single deck to export (by name)",
    )
    am_parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="Keep deck files and notes whose IDs no longer exist in Anki",
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
        "--file",
        "-f",
        help="Single file to import",
    )
    ma_parser.add_argument(
        "--only-add-new",
        action="store_true",
        help="Only add new notes (skip existing notes with note IDs)",
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
        "--no-ids",
        action="store_true",
        help=(
            "Exclude note_key and deck_key from serialized output "
            "(useful for sharing/templates)"
        ),
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
    deserialize_parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Skip writing deck_key and note_key comments (useful for templates/sharing)",
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

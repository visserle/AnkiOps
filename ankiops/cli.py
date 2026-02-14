import argparse
import logging
from pathlib import Path

from ankiops.anki_client import invoke
from ankiops.anki_to_markdown import (
    export_collection,
    export_deck,
)
from ankiops.collection_serializer import (
    serialize_collection_to_json,
    deserialize_collection_from_json,
)
from ankiops.config import get_auto_commit, require_collection_dir
from ankiops.ensure_models import ensure_models
from ankiops.git import git_snapshot
from ankiops.init import create_tutorial, initialize_collection
from ankiops.log import configure_logging, format_changes
from ankiops.markdown_to_anki import (
    import_collection,
    import_file,
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
    media_dir = invoke("getMediaDirPath")

    auto_commit = not args.no_auto_commit
    collection_dir = initialize_collection(profile, media_dir, auto_commit)

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

    if get_auto_commit(collection_dir) and not args.no_auto_commit:
        git_snapshot(collection_dir, "export")

    if args.deck:
        logger.debug(f"Processing deck: {args.deck}...")
        result = export_deck(args.deck, output_dir=str(collection_dir))
        results = [result] if result.file_path else []
        renamed_files = 0
        deleted_decks = 0
        deleted_notes = 0
    else:
        summary = export_collection(
            output_dir=str(collection_dir),
            keep_orphans=args.keep_orphans,
        )
        results = summary.deck_results
        renamed_files = summary.renamed_files
        deleted_decks = summary.deleted_deck_files
        deleted_notes = summary.deleted_orphan_notes

    updated = sum(r.updated for r in results)
    created = sum(r.created for r in results)
    deleted = sum(r.deleted for r in results) + deleted_notes
    total = sum(r.total_notes for r in results)

    changes = format_changes(
        updated=updated,
        created=created,
        deleted=deleted,
        renamed=renamed_files,
        orphaned=deleted_decks,
    )
    logger.info(f"Export complete: {len(results)} files, {total} notes — {changes}")


def run_ma(args):
    """Markdown -> Anki: import markdown files into Anki."""
    connect_or_exit()
    active_profile = invoke("getActiveProfile")
    ensure_models()

    collection_dir = require_collection_dir(active_profile)
    logger.debug(f"Collection directory: {collection_dir}")

    if get_auto_commit(collection_dir) and not args.no_auto_commit:
        git_snapshot(collection_dir, "import")

    deleted_notes = 0

    if args.file:
        logger.debug(f"Processing {Path(args.file).name}...")
        results = [
            import_file(
                Path(args.file),
                only_add_new=args.only_add_new,
            )
        ]
        summary = None
    else:
        summary = import_collection(str(collection_dir), only_add_new=args.only_add_new)
        results = summary.file_results

        # Handle untracked decks (AnkiOps notes in Anki with no markdown file)
        if summary.untracked_decks:
            logger.warning(
                "The following Anki decks with AnkiOps notes inside "
                "have no matching markdown file:"
            )
            for ud in summary.untracked_decks:
                logger.warning(
                    f"  - '{ud.deck_name}' "
                    f"(deck_id: {ud.deck_id}, {len(ud.note_ids)} managed notes)"
                )
            answer = (
                input(
                    "Delete these managed notes from Anki "
                    "(only AnkiOps notes will be removed)? [y/N] "
                )
                .strip()
                .lower()
            )
            if answer == "y":
                for ud in summary.untracked_decks:
                    invoke("deleteNotes", notes=ud.note_ids)
                    deleted_notes += len(ud.note_ids)
                    logger.info(
                        f"Deleted {len(ud.note_ids)} managed notes from "
                        f"'{ud.deck_name}'"
                    )
            else:
                logger.debug("Skipped untracked note deletion.")

    updated = sum(r.updated for r in results)
    created = sum(r.created for r in results)
    deleted = sum(r.deleted for r in results) + deleted_notes
    moved = sum(r.moved for r in results)
    errors = sum(len(r.errors) for r in results)
    total = sum(r.total_notes for r in results)

    changes = format_changes(
        updated=updated,
        created=created,
        deleted=deleted,
        moved=moved,
        errors=errors,
    )
    logger.info(f"Import complete: {len(results)} files, {total} notes — {changes}")
    if errors:
        logger.critical(
            "Review and resolve errors above, then re-run the import — "
            "or you risk losing notes with the next export."
        )


def run_serialize(args):
    """Serialize collection to JSON format."""
    active_profile = invoke("getActiveProfile")
    collection_dir = require_collection_dir(active_profile)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"{collection_dir.name}.json")

    logger.debug(f"Serializing collection from: {collection_dir}")
    logger.debug(f"Output file: {output_file}")

    include_ids = not args.no_ids
    include_media = args.include_media
    serialize_collection_to_json(
        collection_dir,
        output_file,
        include_ids=include_ids,
        include_media=include_media,
    )


def run_deserialize(args):
    """Deserialize collection from JSON/ZIP format."""
    serialized_file = Path(args.serialized_file)

    if not serialized_file.exists():
        logger.error(f"Serialized file not found: {serialized_file}")
        raise SystemExit(1)

    # Determine collection directory
    if args.directory:
        collection_dir = Path(args.directory)
    else:
        # Use filename (without extension) as collection directory name
        collection_dir = Path(serialized_file.stem)

    logger.debug(f"Importing serialized collection from: {serialized_file}")
    logger.debug(f"Target collection directory: {collection_dir}")

    if collection_dir.exists() and not args.overwrite:
        logger.warning(
            f"Collection directory {collection_dir} already exists. "
            "Use --overwrite to replace existing files."
        )

    deserialize_collection_from_json(
        serialized_file, collection_dir, overwrite=args.overwrite
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
        "--no-auto-commit",
        action="store_true",
        help="Disable automatic git commits before sync operations",
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
        help="Export your AnkiOps collection to a portable JSON/ZIP file",
    )
    serialize_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: <collection-name>.json)",
    )
    serialize_parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Exclude note_id and deck_id from serialized output (useful for sharing/templates)",
    )
    serialize_parser.add_argument(
        "--include-media",
        action="store_true",
        help="Bundle media files into a ZIP archive (creates .zip instead of .json)",
    )
    serialize_parser.set_defaults(handler=run_serialize)

    # Deserialize parser
    deserialize_parser = subparsers.add_parser(
        "deserialize",
        help="Import a serialized collection (JSON/ZIP) into a local AnkiOps directory",
    )
    deserialize_parser.add_argument(
        "serialized_file",
        metavar="FILE",
        help="Serialized file to import (.json or .zip)",
    )
    deserialize_parser.add_argument(
        "--directory",
        "-d",
        metavar="DIR",
        help="Local collection directory to create/update (default: use file name)",
    )
    deserialize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files (Anki media uses smart conflict resolution)",
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
            "  init              Initialize current directory as a AnkiOps collection"
        )
        print("  anki-to-markdown  Export Anki decks to Markdown files (alias: am)")
        print("  markdown-to-anki  Import Markdown files into Anki (alias: ma)")
        print("  serialize         Export collection to a portable JSON/ZIP file")
        print("  deserialize       Import a serialized file into a local AnkiOps directory")
        print()
        print("Usage examples:")
        print("  ankiops init --tutorial            # Initialize with tutorial")
        print("  ankiops am                         # Export all decks to Markdown")
        print("  ankiops ma                         # Import all Markdown files to Anki")
        print("  ankiops serialize -o my-deck.json  # Serialize collection to file")
        print("  ankiops deserialize my-deck.json   # Deserialize file to local directory")
        print()
        print("For more information:")
        print("  ankiops --help                 # Show general help")
        print("  ankiops <command> --help       # Show help for a specific command")
        print("=" * 60)


if __name__ == "__main__":
    main()

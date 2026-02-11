import argparse
import logging
from pathlib import Path

from deckops.anki_client import invoke
from deckops.anki_to_markdown import (
    delete_orphaned_cards,
    delete_orphaned_decks,
    rename_markdown_files,
    transcribe_collection,
    transcribe_deck,
)
from deckops.config import get_auto_commit, require_collection_dir
from deckops.ensure_models import ensure_models
from deckops.git import git_snapshot
from deckops.init import create_tutorial, initialize_collection
from deckops.log import configure_logging
from deckops.markdown_to_anki import (
    delete_untracked_notes,
    import_collection,
    import_file,
)

logger = logging.getLogger(__name__)


def connect_or_exit():
    """Verify AnkiConnect is reachable; exit on failure."""
    try:
        version = invoke("version")
        logger.info(f"Connected to AnkiConnect (version {version})")
    except Exception as e:
        logger.error(f"Error connecting to AnkiConnect: {e}")
        logger.error("Make sure Anki is running and AnkiConnect is installed.")
        raise SystemExit(1)


def run_init(args):
    """Initialize the current directory as an DeckOps collection."""
    connect_or_exit()
    profile = invoke("getActiveProfile")
    media_dir = invoke("getMediaDirPath")

    auto_commit = not args.no_auto_commit
    collection_dir = initialize_collection(profile, media_dir, auto_commit)

    if args.tutorial:
        create_tutorial(collection_dir)
        logger.info("Created tutorial file in collection directory")

    logger.info(
        f"Initialized DeckOps collection in {collection_dir} (profile: {profile})"
    )


def run_am(args):
    """Anki -> Markdown: export decks to markdown files."""
    connect_or_exit()
    active_profile = invoke("getActiveProfile")

    collection_dir = require_collection_dir(active_profile)
    logger.info(f"Collection directory: {collection_dir}")

    if get_auto_commit(collection_dir) and not args.no_auto_commit:
        git_snapshot(collection_dir, "export")

    if args.deck:
        logger.info(f"Processing deck: {args.deck}...")
        deck_names_and_ids = invoke("deckNamesAndIds")
        deck_id = deck_names_and_ids.get(args.deck)
        result = transcribe_deck(
            args.deck, output_dir=str(collection_dir), deck_id=deck_id
        )
        results = [result] if result.file_path else []
        renamed_files = 0
    else:
        renamed_files = rename_markdown_files(output_dir=str(collection_dir))
        results = transcribe_collection(output_dir=str(collection_dir))

    total = sum(r.total_notes for r in results)
    updated = sum(r.updated for r in results)
    created = sum(r.created for r in results)
    deleted = sum(r.deleted for r in results)
    skipped = sum(r.skipped for r in results)

    if not args.keep_orphans and not args.deck:
        deleted_decks = delete_orphaned_decks(output_dir=str(collection_dir))
        deleted_cards = delete_orphaned_cards(output_dir=str(collection_dir))
    else:
        deleted_decks = 0
        deleted_cards = 0

    logger.info(f"{'=' * 60}")
    logger.info(f"Export complete: {len(results)} files processed")
    logger.info(f"Total notes: {total}")
    logger.info(
        f"Updated: {updated}, Created: {created}, "
        f"Deleted: {deleted + deleted_cards}, Skipped: {skipped}"
    )
    if renamed_files:
        logger.info(f"Renamed: {renamed_files} deck file(s)")
    if deleted_decks:
        logger.info(f"Deleted: {deleted_decks} orphaned deck file(s)")


def run_ma(args):
    """Markdown -> Anki: import markdown files into Anki."""
    connect_or_exit()
    active_profile = invoke("getActiveProfile")
    ensure_models()

    collection_dir = require_collection_dir(active_profile)
    logger.info(f"Collection directory: {collection_dir}")

    if get_auto_commit(collection_dir) and not args.no_auto_commit:
        git_snapshot(collection_dir, "import")

    deleted_notes = 0

    if args.file:
        logger.info(f"Processing {Path(args.file).name}...")
        results = [
            import_file(
                Path(args.file),
                only_add_new=args.only_add_new,
            )
        ]
    else:
        results = import_collection(str(collection_dir), only_add_new=args.only_add_new)
        deleted_notes = delete_untracked_notes(collection_dir=str(collection_dir))

    total = sum(r.total_notes for r in results)
    updated = sum(r.updated for r in results)
    created = sum(r.created for r in results)
    deleted = sum(r.deleted for r in results)
    skipped = sum(r.skipped for r in results)
    errors = sum(len(r.errors) for r in results)

    logger.info(f"{'=' * 60}")
    logger.info(f"Import complete: {len(results)} files processed")
    logger.info(f"Total notes: {total}")
    logger.info(
        f"Updated: {updated}, Created: {created}, "
        f"Deleted: {deleted}, Skipped: {skipped}, "
        f"Errors: {errors}"
    )
    if deleted_notes:
        logger.info(f"Deleted: {deleted_notes} managed note(s) from untracked decks")
    if errors:
        logger.warning(
            "Errors occurred during import. Review and resolve them to ensure "
            "all cards remain properly tracked in future exports."
        )


def main():
    parser = argparse.ArgumentParser(
        description="DeckOps – Manage Anki decks as Markdown files.",
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
        help="Initialize current directory as an DeckOps collection",
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
        help="Keep deck files and cards whose IDs no longer exist in Anki",
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
        help="Only add new cards (skip existing cards with card or note IDs)",
    )
    ma_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    ma_parser.set_defaults(handler=run_ma)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(stream_level=log_level, ignore_libs=["urllib3.connectionpool"])

    if hasattr(args, "handler"):
        args.handler(args)
    else:
        # Show welcome screen when no command is provided
        print("=" * 60)
        print("DeckOps – Manage Anki decks as Markdown files")
        print("=" * 60)
        print()
        print("Available commands:")
        print(
            "  init              Initialize current directory as a DeckOps collection"
        )
        print("  anki-to-markdown  Export Anki decks to Markdown files (alias: am)")
        print("  markdown-to-anki  Import Markdown files into Anki (alias: ma)")
        print()
        print("Usage examples:")
        print("  deckops init --tutorial        # Initialize with tutorial")
        print("  deckops am                     # Export all decks to Markdown")
        print("  deckops ma                     # Import all Markdown files to Anki")
        print()
        print("For more information:")
        print("  deckops --help                 # Show general help")
        print("  deckops <command> --help       # Show help for a specific command")
        print("=" * 60)


if __name__ == "__main__":
    main()

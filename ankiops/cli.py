import argparse
import logging
from importlib.metadata import PackageNotFoundError, version

from ankiops.anki_rpc import AnkiConnectionError
from ankiops.cli_commands import (
    run_af,
    run_deserialize,
    run_fa,
    run_fix_image_widths,
    run_init,
    run_llm,
    run_note_type,
    run_serialize,
    run_shared,
)
from ankiops.console import configure_logging
from ankiops.llm.commands import configure_llm_parser

logger = logging.getLogger(__name__)


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


def _single_line_text(value: str) -> str:
    parsed = value.strip()
    if not parsed or "\n" in parsed or "\r" in parsed:
        raise argparse.ArgumentTypeError("must be a non-empty single line")
    return parsed


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

    # Anki to files (af) parser
    af_parser = subparsers.add_parser(
        "anki-to-files",
        aliases=["af"],
        help="Anki -> files",
    )
    af_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    af_parser.set_defaults(handler=run_af)

    # Files to Anki (fa) parser
    fa_parser = subparsers.add_parser(
        "files-to-anki",
        aliases=["fa"],
        help="Files -> Anki",
    )
    fa_parser.add_argument(
        "--no-auto-commit",
        "-n",
        action="store_true",
        help="Skip the automatic git commit for this operation",
    )
    fa_parser.set_defaults(handler=run_fa)

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

    shared_parser = subparsers.add_parser(
        "shared",
        help="Publish, subscribe to, update, and contribute to shared decks",
    )
    shared_subparsers = shared_parser.add_subparsers(
        dest="shared_command",
        required=True,
    )

    shared_publish = shared_subparsers.add_parser(
        "publish",
        help="Publish an existing local deck as a new GitHub repository",
    )
    shared_publish.add_argument("deck", help="Deck to publish (includes subdecks)")
    shared_publish.add_argument(
        "source_id",
        metavar="OWNER/REPO",
        help="Shared deck identity (letters, digits, hyphens)",
    )
    publish_visibility = (
        shared_publish.add_mutually_exclusive_group()
    )  # todo: remove this, public is the default for sharing decks
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
    shared_publish.set_defaults(public=False)
    shared_publish.set_defaults(handler=run_shared)

    shared_subscribe = shared_subparsers.add_parser(
        "subscribe",
        help="Subscribe to a shared deck on GitHub",
    )
    shared_subscribe.add_argument(
        "source_id",
        metavar="OWNER/REPO",
        help="Shared deck identity (letters, digits, hyphens)",
    )
    shared_subscribe.set_defaults(handler=run_shared)

    shared_update = shared_subparsers.add_parser(
        "update",
        help="Bring available GitHub changes into shared Markdown files",
    )
    shared_update.add_argument(
        "source_id",
        metavar="OWNER/REPO",
        nargs="?",
        help="Shared deck identity (letters, digits, hyphens)",
    )
    shared_update.set_defaults(handler=run_shared)

    shared_submit = shared_subparsers.add_parser(
        "submit",
        help="Submit a contribution as a GitHub pull request",
    )
    shared_submit.add_argument(
        "source_id",
        metavar="OWNER/REPO",
        help="Shared deck identity (letters, digits, hyphens)",
    )
    shared_submit.add_argument(
        "--message",
        "-m",
        type=_single_line_text,
        help="Pull request title and commit message for new changes",
    )
    shared_submit.set_defaults(handler=run_shared)

    shared_status = shared_subparsers.add_parser(
        "status",
        help="Preview changes, updates, submissions, and recovery state",
    )
    shared_status.add_argument(
        "source_id",
        metavar="OWNER/REPO",
        nargs="?",
        help="Shared deck identity (letters, digits, hyphens)",
    )
    shared_status.set_defaults(handler=run_shared)

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
        print(f"AnkiOps v{cli_version} – A bidirectional Anki-files bridge")
        print("=" * 60)
        print()
        print("Available commands:")
        print(
            "  init              Initialize current directory as an AnkiOps collection"
        )
        print("  anki-to-files     Sync Anki changes into files (alias: af)")
        print("  files-to-anki     Sync file changes into Anki (alias: fa)")
        print("  serialize         Serialize Markdown decks to JSON format")
        print("  deserialize       Deserialize JSON file to Markdown decks")
        print("  fix-image-widths  Normalize or force Markdown image widths")
        print("  llm               Status/plan/run LLM jobs and inspect one LLM job")
        print(
            "  shared            Publish, subscribe to, update, and contribute to decks"
        )
        print("  note-types        List note type labels or add note types from Anki")
        print()
        print("Usage examples:")
        print(
            "  ankiops init --tutorial                      # Initialize with tutorial"
        )
        print("  ankiops af                                   # Sync Anki to files")
        print("  ankiops fa                                   # Sync files to Anki")
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
        print("  ankiops shared status                        # Check shared decks")
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

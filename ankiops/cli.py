import argparse
import logging
from pathlib import Path

from ankiops.ai import (
    AIConfigError,
    AIPaths,
    AIRequestError,
    AIResponseError,
    AIRuntimeOverrides,
    OpenAICompatibleAsyncEditor,
    PromptConfigError,
    PromptExecutionError,
    PromptRunner,
    PromptRunOptions,
    prepare_ai_run,
)
from ankiops.ai.model_profiles import load_model_profiles, resolve_runtime_config
from ankiops.anki import AnkiAdapter
from ankiops.collection_serializer import (
    deserialize_collection_data,
    deserialize_collection_from_json,
    serialize_collection,
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
from ankiops.models import CollectionResult
from ankiops.sync_media import sync_media_from_anki, sync_media_to_anki
from ankiops.sync_note_types import sync_note_types

logger = logging.getLogger(__name__)


def require_initialized_collection_dir() -> Path:
    """Return collection directory or exit if no local AnkiOps DB exists."""
    collection_dir = get_collection_dir()
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        logger.error(
            f"Not an AnkiOps collection ({collection_dir}). Run 'ankiops init' first."
        )
        raise SystemExit(1)
    return collection_dir


def connect_or_exit() -> AnkiAdapter:
    """Verify AnkiConnect is reachable; exit on failure. Returns the adapter."""
    anki = AnkiAdapter()
    try:
        version = anki.get_version()
        logger.debug(f"Connected to AnkiConnect (version {version})")
    except Exception as error:
        logger.error(f"Error connecting to AnkiConnect: {error}")
        logger.error("Make sure Anki is running and AnkiConnect is installed.")
        raise SystemExit(1)
    return anki


def _format_media_status(media_result, *, from_anki: bool) -> str:
    checked = media_result.checked
    summary = media_result.summary

    if checked == 0:
        return "Media: no referenced files"

    if from_anki and media_result.missing:
        return (
            f"Media: {checked} files checked — "
            f"{summary.synced} pulled, {media_result.missing} missing in Anki"
        )

    return f"Media: {checked} files checked — {summary.format()}"


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
    db = SQLiteDbAdapter.load(collection_dir)
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
    db = SQLiteDbAdapter.load(collection_dir)
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
    )


def run_deserialize(args):
    """Deserialize collection from JSON/ZIP format to target directory."""
    serialized_file = Path(args.serialized_file)

    if not serialized_file.exists():
        logger.error(f"Serialized file not found: {serialized_file}")
        raise SystemExit(1)

    deserialize_collection_from_json(
        serialized_file,
        overwrite=args.overwrite,
    )


def _runtime_overrides_from_args(args) -> AIRuntimeOverrides:
    return AIRuntimeOverrides(
        profile=args.profile,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        max_in_flight=args.max_in_flight,
        api_key=args.api_key,
    )


def run_ai_config(args):
    """Show the resolved AI runtime configuration from ai/models.yaml."""
    collection_dir = require_initialized_collection_dir()
    ai_paths = AIPaths.from_collection_dir(collection_dir)

    try:
        models_config = load_model_profiles(ai_paths)
    except AIConfigError as error:
        logger.error(f"Invalid model profile configuration: {error}")
        raise SystemExit(1)

    overrides = _runtime_overrides_from_args(args)
    try:
        runtime = resolve_runtime_config(
            models_config,
            profile=overrides.profile,
            provider=overrides.provider,
            model=overrides.model,
            base_url=overrides.base_url,
            api_key_env=overrides.api_key_env,
            timeout_seconds=overrides.timeout_seconds,
            max_in_flight=overrides.max_in_flight,
            api_key=overrides.api_key,
        )
    except AIConfigError as error:
        logger.error(f"Invalid AI configuration: {error}")
        raise SystemExit(1)

    logger.info(f"Models config: {models_config.source_path}")
    logger.info(f"Default profile: {models_config.default_profile}")
    logger.info(f"Selected profile: {runtime.profile}")
    logger.info(f"AI provider: {runtime.provider}")
    logger.info(f"AI model: {runtime.model}")
    logger.info(f"AI base URL: {runtime.base_url}")
    logger.info(f"AI timeout: {runtime.timeout_seconds}s")
    logger.info(f"AI max in flight: {runtime.max_in_flight}")
    logger.info(f"API key env var: {runtime.api_key_env}")
    logger.info(f"API key available: {'yes' if runtime.api_key else 'no'}")


def run_ai_prompt(args):
    """Run prompt-driven inline JSON edits over serialized collection data."""
    collection_dir = require_initialized_collection_dir()

    if not args.prompt:
        logger.error("Missing required argument: --prompt")
        raise SystemExit(2)
    if args.batch_size <= 0:
        logger.error("--batch-size must be > 0")
        raise SystemExit(2)

    overrides = _runtime_overrides_from_args(args)
    try:
        prompt_config, runtime = prepare_ai_run(
            collection_dir=collection_dir,
            prompt_ref=args.prompt,
            overrides=overrides,
        )
    except PromptConfigError as error:
        logger.error(f"Invalid prompt configuration: {error}")
        raise SystemExit(1)
    except AIConfigError as error:
        logger.error(f"Invalid AI configuration: {error}")
        raise SystemExit(1)

    if runtime.provider == "remote" and not runtime.api_key:
        logger.error(
            f"No API key found in env var '{runtime.api_key_env}'. "
            "Set it or pass --api-key."
        )
        raise SystemExit(1)

    logger.info(
        "AI prompt run "
        f"prompt='{prompt_config.name}' "
        f"profile='{runtime.profile}' "
        f"provider='{runtime.provider}' model='{runtime.model}' "
        f"batch_size={args.batch_size} max_in_flight={runtime.max_in_flight}"
    )

    serialized_data = serialize_collection(collection_dir)

    client = OpenAICompatibleAsyncEditor(runtime)
    options = PromptRunOptions(
        include_decks=args.include_deck,
        batch_size=args.batch_size,
        max_in_flight=runtime.max_in_flight,
    )
    try:
        result = PromptRunner(client).run(
            serialized_data=serialized_data,
            prompt=prompt_config,
            options=options,
        )
    except (PromptExecutionError, AIRequestError, AIResponseError) as error:
        logger.error(f"AI prompt failed: {error}")
        raise SystemExit(1)

    if args.include_deck and result.processed_decks == 0:
        logger.warning("No deck matched --include-deck filters.")
        return

    logger.info(
        "AI prompt processed "
        f"{result.prompted_notes} prompted note(s), "
        f"{result.processed_notes} scanned note(s), "
        f"across {result.processed_decks} deck(s)."
    )
    logger.info(f"AI prompt changed {result.changed_fields} field(s).")

    for change in result.changes[:20]:
        logger.info(
            f"  {change.deck_name} [{change.note_key or 'new'}] {change.field_name}"
        )
    if len(result.changes) > 20:
        logger.info(f"  ... and {len(result.changes) - 20} more change(s)")

    for warning in result.warnings[:20]:
        logger.warning(warning)
    remaining_warnings = max(0, len(result.warnings) - 20) + result.dropped_warnings
    if remaining_warnings:
        logger.warning(f"... and {remaining_warnings} more warning(s)")

    if result.changed_fields == 0:
        logger.info("No changes to write.")
        return

    apply_payload = {
        "collection": serialized_data.get("collection", {}),
        "decks": result.changed_decks,
    }

    deserialize_collection_data(
        apply_payload,
        overwrite=True,
    )

    logger.info(f"Applied changes to {len(result.changed_decks)} deck(s).")


def add_ai_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        help="Model profile name from ai/models.yaml",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "remote"],
        help="Override provider for selected profile",
    )
    parser.add_argument(
        "--model",
        help="Override model name",
    )
    parser.add_argument(
        "--base-url",
        help="Override OpenAI-compatible chat completions base URL",
    )
    parser.add_argument(
        "--api-key-env",
        help="Override environment variable name used for API key lookup",
    )
    parser.add_argument(
        "--api-key",
        help="API key value (runtime only; never persisted)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Override request timeout in seconds",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="Override maximum concurrent AI requests",
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

    # AI parser
    ai_parser = subparsers.add_parser(
        "ai",
        help="Run prompt-driven AI edits on serialized collection data",
    )
    ai_parser.set_defaults(handler=run_ai_prompt)
    ai_parser.add_argument(
        "--include-deck",
        "-d",
        action="append",
        default=[],
        help=(
            "Deck name to include; always includes all subdecks recursively. "
            "Repeat to include multiple deck trees."
        ),
    )
    ai_parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt file name/path from ai/prompts/ (or absolute path)",
    )
    ai_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of notes to send per model request (default: 1)",
    )
    add_ai_runtime_args(ai_parser)

    ai_subparsers = ai_parser.add_subparsers(dest="ai_command", required=False)

    ai_config_parser = ai_subparsers.add_parser(
        "config",
        help="Show resolved runtime config from ai/models.yaml",
    )
    add_ai_runtime_args(ai_config_parser)
    ai_config_parser.set_defaults(handler=run_ai_config)

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
        print("  ai                Run prompt-driven AI edits (or ai config)")
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
            "  ankiops deserialize my-deck.json             "
            "# Deserialize file, then run init"
        )
        print(
            "  ankiops ai --prompt grammar --profile local-fast -d Biology"
            "  # Prompt-run a deck tree"
        )
        print(
            "  ankiops ai config --profile remote-fast      # Show runtime model config"
        )
        print()
        print("For more information:")
        print("  ankiops --help                 # Show general help")
        print("  ankiops <command> --help       # Show help for a specific command")
        print("=" * 60)


if __name__ == "__main__":
    main()

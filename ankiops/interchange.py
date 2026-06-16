"""Serialize and deserialize AnkiOps decks to/from JSON format."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ankiops.collection import (
    ANKIOPS_DB,
    NOTE_TYPES_DIR,
    deck_name_in_scope,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
    get_collection_dir,
)
from ankiops.console import clickable_path
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
)
from ankiops.markdown import (
    DeckFile,
    read_deck_file,
    render_notes_to_markdown,
    write_deck_file,
)
from ankiops.note_types import NoteType
from ankiops.notes import Note

logger = logging.getLogger(__name__)

LOCAL_SOURCE_NAME = "local"


@dataclass(frozen=True)
class _ParsedDeck:
    source: DeckSource
    source_name: str
    path: Path
    deck_name: str
    parsed: DeckFile


@dataclass(frozen=True)
class _ValidatedDeck:
    source: DeckSource
    source_name: str
    name: str
    notes: list[dict[str, Any]]
    note_types_by_name: dict[str, NoteType]


@dataclass(frozen=True)
class DeserializationPlan:
    decks: tuple[_ValidatedDeck, ...]
    target_paths: tuple[Path, ...]
    has_shared_sources: bool


def _source_name(source: DeckSource) -> str:
    return source.source_id or LOCAL_SOURCE_NAME


def serialize(
    collection_dir: Path,
    *,
    deck: str | None = None,
    no_subdecks: bool = False,
    note_types_dir: Path | None = None,
) -> dict[str, Any]:
    """Serialize markdown decks into an in-memory JSON-compatible mapping."""
    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an AnkiOps collection: {collection_dir}")

    sources = discover_deck_sources(collection_dir, note_types_dir=note_types_dir)
    parsed_decks = _parse_sources(collection_dir, sources)
    _validate_parsed_decks(parsed_decks)

    serialized_data: dict[str, Any] = {
        "collection": {
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    deck_filter = deck.strip() if isinstance(deck, str) else None
    filtered_decks = [
        parsed_deck
        for parsed_deck in parsed_decks
        if deck_name_in_scope(
            parsed_deck.deck_name,
            deck=deck_filter,
            no_subdecks=no_subdecks,
        )
    ]

    for parsed_deck in filtered_decks:
        notes_data = [
            {
                "note_key": note.note_key,
                "note_type": note.note_type,
                "fields": note.fields,
                "tags": list(note.tags),
            }
            for note in parsed_deck.parsed.notes
        ]
        if not notes_data:
            continue
        decks_data = serialized_data["decks"]
        if isinstance(decks_data, list):
            decks_data.append(
                {
                    "source": parsed_deck.source_name,
                    "name": parsed_deck.deck_name,
                    "notes": notes_data,
                }
            )

    total_notes = sum(
        len(deck.get("notes", []))
        for deck in serialized_data["decks"]
        if isinstance(deck, dict)
    )
    total_decks = len(serialized_data["decks"])
    logger.debug(
        "Serialized %d deck(s), %d note(s), %d source(s) to in-memory mapping",
        total_decks,
        total_notes,
        len(sources),
    )

    return serialized_data


def _parse_sources(
    collection_dir: Path,
    sources: list[DeckSource],
) -> list[_ParsedDeck]:
    parsed_decks: list[_ParsedDeck] = []
    for source in sources:
        configs = load_note_types_for_source(source)
        for md_file in source.deck_files():
            try:
                parsed = read_deck_file(
                    md_file,
                    note_types=configs,
                    context_root=source.root,
                )
            except Exception as error:
                display = _display_source_path(collection_dir, source, md_file)
                raise ValueError(f"Error parsing {display}: {error}") from error
            parsed_decks.append(
                _ParsedDeck(
                    source=source,
                    source_name=_source_name(source),
                    path=md_file,
                    deck_name=file_stem_to_deck_name(md_file.stem),
                    parsed=parsed,
                )
            )
    parsed_decks.sort(
        key=lambda item: (
            0 if item.source_name == LOCAL_SOURCE_NAME else 1,
            item.source_name,
            item.deck_name,
        )
    )
    return parsed_decks


def _validate_parsed_decks(parsed_decks: list[_ParsedDeck]) -> None:
    errors: list[str] = []
    deck_by_name: dict[str, _ParsedDeck] = {}
    note_key_by_value: dict[str, tuple[_ParsedDeck, int]] = {}

    for parsed_deck in parsed_decks:
        previous_deck = deck_by_name.get(parsed_deck.deck_name)
        if previous_deck is not None:
            errors.append(
                "Duplicate deck name "
                f"'{parsed_deck.deck_name}' in "
                f"{previous_deck.source_name}:{previous_deck.path.name} and "
                f"{parsed_deck.source_name}:{parsed_deck.path.name}."
            )
        else:
            deck_by_name[parsed_deck.deck_name] = parsed_deck

        for index, note in enumerate(parsed_deck.parsed.notes, start=1):
            note_key = note.note_key
            if note_key is not None:
                previous_note = note_key_by_value.get(note_key)
                if previous_note is not None:
                    previous_deck, previous_index = previous_note
                    errors.append(
                        "Duplicate note_key "
                        f"'{note_key}' in "
                        f"{previous_deck.source_name}:{previous_deck.path.name} "
                        f"note {previous_index} and "
                        f"{parsed_deck.source_name}:{parsed_deck.path.name} "
                        f"note {index}."
                    )
                else:
                    note_key_by_value[note_key] = (parsed_deck, index)

    if errors:
        raise ValueError("Cannot serialize collection:\n- " + "\n- ".join(errors))


def _display_source_path(collection_dir: Path, source: DeckSource, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(source.root.resolve())
    except ValueError:
        try:
            relative = path.resolve().relative_to(collection_dir.resolve())
        except ValueError:
            return str(path)
    return f"{_source_name(source)}:{relative}"


def serialize_to_file(
    collection_dir: Path,
    output_file: Path,
    *,
    deck: str | None = None,
    no_subdecks: bool = False,
    note_types_dir: Path | None = None,
) -> dict[str, Any]:
    """Serialize markdown decks to a JSON file.

    Args:
        collection_dir: Path to the markdown decks directory
        output_file: Path where JSON file will be written
        deck: Optional deck name scope for serialization, otherswise serializes the
              entire collection
        no_subdecks: If True with deck set, include only exact deck

    Returns:
        Dictionary containing the serialized data
    """
    serialized_data = serialize(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=note_types_dir,
    )
    total_notes = sum(
        len(deck.get("notes", []))
        for deck in serialized_data["decks"]
        if isinstance(deck, dict)
    )
    total_decks = len(serialized_data["decks"])

    with output_file.open("w", encoding="utf-8") as output_handle:
        json.dump(serialized_data, output_handle, indent=2, ensure_ascii=False)

    logger.info(
        f"Serialized {total_decks} deck(s), {total_notes} note(s) to {output_file}"
    )

    return serialized_data


def deserialize(
    data: dict[str, Any],
    *,
    collection_dir: Path,
    note_types_dir: Path,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Deserialize markdown decks from an in-memory JSON-compatible mapping."""
    logger.debug(f"Target directory: {collection_dir}")

    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an initialized AnkiOps collection: {collection_dir}")

    decks = _validate_serialized_data(
        data,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
    )
    _write_validated_decks(
        decks,
        collection_dir=collection_dir,
        overwrite=overwrite,
        quiet=quiet,
    )


def apply_deserialization_plan(
    plan: DeserializationPlan,
    *,
    collection_dir: Path,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Write a previously validated deserialization plan."""
    logger.debug(f"Target directory: {collection_dir}")

    db_path = collection_dir / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an initialized AnkiOps collection: {collection_dir}")

    _write_validated_decks(
        plan.decks,
        collection_dir=collection_dir,
        overwrite=overwrite,
        quiet=quiet,
    )


def _write_validated_decks(
    decks: Sequence[_ValidatedDeck],
    *,
    collection_dir: Path,
    overwrite: bool,
    quiet: bool,
) -> None:
    total_decks = 0
    total_notes = 0

    for deck in decks:
        output_path = _deserialize_target_path(deck)
        notes = [
            Note(
                note_key=cast(str | None, note["note_key"]),
                note_type=cast(str, note["note_type"]),
                fields=cast(dict[str, str], note["fields"]),
                tags=cast(list[str], note["tags"]),
            )
            for note in deck.notes
        ]
        content = (
            render_notes_to_markdown(notes, deck.note_types_by_name) if notes else ""
        )
        written_notes = len(notes)
        if overwrite or not output_path.exists():
            write_deck_file(output_path, content)
            created_message = (
                f"  Created {clickable_path(output_path)} ({written_notes} notes)"
            )
            if quiet:
                logger.debug(created_message, extra={"markup": True})
            else:
                logger.info(created_message, extra={"markup": True})
        else:
            logger.warning(
                f"Skipped {clickable_path(output_path)} "
                "(already exists, use --overwrite to replace)",
                extra={"markup": True},
            )

        total_decks += 1
        total_notes += written_notes

    summary_message = (
        "Deserialized "
        f"{total_decks} deck(s), {total_notes} note(s), "
        f"{len({deck.source_name for deck in decks})} source(s) to {collection_dir}"
    )
    if quiet:
        logger.debug(summary_message)
    else:
        logger.info(summary_message)


def _deserialize_target_path(deck: _ValidatedDeck) -> Path:
    return deck.source.root / (deck_name_to_file_stem(deck.name) + ".md")


def _validate_serialized_data(
    data: dict[str, Any],
    *,
    collection_dir: Path,
    note_types_dir: Path,
) -> list[_ValidatedDeck]:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("Serialized data must be a JSON object mapping")

    raw_decks = data.get("decks")
    if not isinstance(raw_decks, list):
        raise ValueError("Serialized data must contain a top-level 'decks' list")

    sources = discover_deck_sources(collection_dir, note_types_dir=note_types_dir)
    source_by_name = {_source_name(source): source for source in sources}
    configs_by_source: dict[str, dict[str, NoteType]] = {}
    referenced_sources = {
        deck.get("source")
        for deck in raw_decks
        if isinstance(deck, dict) and isinstance(deck.get("source"), str)
    }
    for source_name in sorted(referenced_sources):
        source = source_by_name.get(source_name)
        if source is None:
            continue
        try:
            configs = load_note_types_for_source(source)
        except Exception as error:
            errors.append(
                f"Source '{source_name}' note_types/ cannot be loaded: {error}"
            )
            continue
        configs_by_source[source_name] = {config.name: config for config in configs}

    seen_decks: dict[str, int] = {}
    seen_note_keys: dict[str, tuple[str, int, int]] = {}
    validated: list[_ValidatedDeck] = []

    for deck_index, raw_deck in enumerate(raw_decks, start=1):
        if not isinstance(raw_deck, dict):
            errors.append(f"Deck {deck_index} must be an object.")
            continue

        source_name = raw_deck.get("source")
        deck_name = raw_deck.get("name")
        raw_notes = raw_deck.get("notes")
        if not isinstance(source_name, str) or not source_name.strip():
            errors.append(f"Deck {deck_index} is missing required source.")
            source_name = ""
        if not isinstance(deck_name, str) or not deck_name.strip():
            errors.append(f"Deck {deck_index} is missing required name.")
            deck_name = ""
        if not isinstance(raw_notes, list):
            errors.append(f"Deck {deck_index} is missing required notes list.")
            raw_notes = []

        source = source_by_name.get(source_name)
        if source_name and source is None:
            errors.append(
                f"Deck '{deck_name or deck_index}' references unknown source "
                f"'{source_name}'."
            )
        configs = configs_by_source.get(source_name, {})

        if deck_name:
            previous_index = seen_decks.get(deck_name)
            if previous_index is not None:
                errors.append(
                    f"Duplicate deck name '{deck_name}' in deck {previous_index} "
                    f"and deck {deck_index}."
                )
            else:
                seen_decks[deck_name] = deck_index

        notes: list[dict[str, Any]] = []
        for note_index, raw_note in enumerate(raw_notes, start=1):
            context = f"Deck '{deck_name or deck_index}' note {note_index}"
            if not isinstance(raw_note, dict):
                errors.append(f"{context} must be an object.")
                continue

            raw_note_key = raw_note.get("note_key")
            note_type = raw_note.get("note_type")
            fields = raw_note.get("fields")
            tags = raw_note.get("tags")
            note_errors_start = len(errors)

            note_key: str | None = None
            if "note_key" not in raw_note:
                errors.append(f"{context} is missing required note_key field.")
            elif raw_note_key is None:
                pass
            elif isinstance(raw_note_key, str) and raw_note_key.strip():
                note_key = raw_note_key
                previous = seen_note_keys.get(note_key)
                if previous is not None:
                    previous_deck, previous_deck_index, previous_note_index = previous
                    errors.append(
                        "Duplicate note_key "
                        f"'{note_key}' in deck {previous_deck_index} "
                        f"('{previous_deck}') note {previous_note_index} and "
                        f"deck {deck_index} ('{deck_name}') note {note_index}."
                    )
                else:
                    seen_note_keys[note_key] = (deck_name, deck_index, note_index)
            else:
                errors.append(f"{context} note_key must be a non-empty string or null.")

            if not isinstance(note_type, str) or not note_type.strip():
                errors.append(f"{context} is missing required note_type.")
            elif source is not None and note_type not in configs:
                errors.append(
                    f"{context} references unknown note_type '{note_type}' "
                    f"for source '{source_name}'."
                )

            if not isinstance(fields, dict):
                errors.append(f"{context} fields must be an object.")
                fields = {}
            else:
                for field_name, field_value in fields.items():
                    if not isinstance(field_name, str) or not field_name.strip():
                        errors.append(
                            f"{context} field names must be non-empty strings."
                        )
                    if not isinstance(field_value, str):
                        errors.append(
                            f"{context} field '{field_name}' must be a string."
                        )

            if not isinstance(tags, list):
                errors.append(f"{context} tags must be a list of strings.")
                tags = []
            else:
                for tag in tags:
                    if not isinstance(tag, str):
                        errors.append(f"{context} tags must contain only strings.")

            if len(errors) == note_errors_start:
                notes.append(
                    {
                        "note_key": note_key,
                        "note_type": note_type,
                        "fields": dict(fields),
                        "tags": list(tags),
                    }
                )

        if (
            source is not None
            and source_name
            and deck_name
            and isinstance(raw_notes, list)
        ):
            validated.append(
                _ValidatedDeck(
                    source=source,
                    source_name=source_name,
                    name=deck_name,
                    notes=notes,
                    note_types_by_name=configs,
                )
            )

    if errors:
        raise ValueError("Cannot deserialize collection:\n- " + "\n- ".join(errors))
    return validated


def plan_deserialize_from_file(
    json_file: Path,
    *,
    collection_dir: Path,
    note_types_dir: Path,
) -> DeserializationPlan:
    with json_file.open("r", encoding="utf-8") as input_handle:
        data = json.load(input_handle)

    logger.debug(f"Importing serialized data from: {json_file}")
    decks = _validate_serialized_data(
        data,
        collection_dir=collection_dir,
        note_types_dir=note_types_dir,
    )
    return DeserializationPlan(
        decks=tuple(decks),
        target_paths=tuple(_deserialize_target_path(deck) for deck in decks),
        has_shared_sources=any(deck.source.is_shared for deck in decks),
    )


def deserialize_from_file(
    json_file: Path,
    overwrite: bool = False,
    *,
    collection_dir: Path | None = None,
    note_types_dir: Path | None = None,
) -> None:
    """Deserialize markdown decks from a JSON file.

    In development mode (pyproject.toml with name="ankiops" in cwd),
    unpacks to ./collection. Otherwise, unpacks to the current working directory.

    Args:
        json_file: Path to JSON file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
    """
    collection_dir = collection_dir or get_collection_dir()
    resolved_note_types_dir = note_types_dir or (collection_dir / NOTE_TYPES_DIR)
    plan = plan_deserialize_from_file(
        json_file,
        collection_dir=collection_dir,
        note_types_dir=resolved_note_types_dir,
    )
    apply_deserialization_plan(
        plan,
        collection_dir=collection_dir,
        overwrite=overwrite,
    )

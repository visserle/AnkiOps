"""Serialize and deserialize AnkiOps decks to/from JSON format."""

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ankiops.collection import (
    ANKIOPS_DB,
    deck_name_to_file_stem,
    file_stem_to_deck_name,
    get_collection_root,
)
from ankiops.console import clickable_path
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_source,
    parse_github_slug,
)
from ankiops.markdown import (
    DeckFile,
    read_deck_file,
    render_notes_to_markdown,
    write_deck_file,
)
from ankiops.media import extract_media_references
from ankiops.note_types import NoteType
from ankiops.notes import Note

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedDeck:
    path: Path
    deck_name: str
    parsed: DeckFile


@dataclass(frozen=True)
class ParsedSource:
    source: DeckSource
    note_types: tuple[NoteType, ...]
    decks: tuple[ParsedDeck, ...]
    applicable_paths: frozenset[str]


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
    has_collab_sources: bool


def _source_name(source: DeckSource) -> str:
    return source.display_name


def serialize(
    collection_root: Path,
    *,
    deck: str | None = None,
    no_subdecks: bool = False,
) -> dict[str, Any]:
    """Serialize markdown decks into an in-memory JSON-compatible mapping."""
    db_path = collection_root / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an AnkiOps collection: {collection_root}")

    parsed_sources = parse_collection(collection_root)
    serialized_data: dict[str, Any] = {
        "collection": {
            "serialized_at": datetime.now(timezone.utc).isoformat(),
        },
        "decks": [],
    }

    deck_filter = deck.strip() if isinstance(deck, str) else None
    filtered_decks = [
        (parsed_source, parsed_deck)
        for parsed_source in parsed_sources
        for parsed_deck in parsed_source.decks
        if _deck_matches_filter(
            parsed_deck.deck_name,
            deck_filter=deck_filter,
            no_subdecks=no_subdecks,
        )
    ]

    for parsed_source, parsed_deck in filtered_decks:
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
                    "source": parsed_source.source.display_name,
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
        len(parsed_sources),
    )

    return serialized_data


def parse_source(source: DeckSource) -> ParsedSource:
    parsed_source = _parse_source(source)
    _validate_parsed_sources((parsed_source,))
    return parsed_source


def _parse_source(source: DeckSource) -> ParsedSource:
    note_types = load_note_types_for_source(source)
    note_types_by_name = {note_type.name: note_type for note_type in note_types}
    decks = []
    applicable_paths: set[str] = set()
    used_note_types: set[str] = set()
    for md_file in source.deck_files():
        try:
            parsed = read_deck_file(
                md_file,
                note_types=note_types,
                context_root=source.root,
            )
        except Exception as error:
            _raise_source_parse_error(source, md_file, error)
        decks.append(
            ParsedDeck(
                path=md_file,
                deck_name=file_stem_to_deck_name(md_file.stem),
                parsed=parsed,
            )
        )
        applicable_paths.add(md_file.relative_to(source.root).as_posix())
        applicable_paths.update(
            f"media/{filename}"
            for filename in extract_media_references(parsed.raw_content)
        )
        used_note_types.update(note.note_type for note in parsed.notes)

    for name in used_note_types:
        applicable_paths.update(
            (Path("note_types") / path).as_posix()
            for path in note_types_by_name[name].source_files
        )
    return ParsedSource(
        source=source,
        note_types=tuple(note_types),
        decks=tuple(decks),
        applicable_paths=frozenset(applicable_paths),
    )


def parse_collection(collection_root: Path) -> tuple[ParsedSource, ...]:
    parsed_sources = tuple(
        _parse_source(source) for source in discover_deck_sources(collection_root)
    )
    _validate_parsed_sources(parsed_sources)
    return parsed_sources


def require_note_keys(deck_files: Iterable[DeckFile]) -> set[str]:
    note_key_locations: dict[str, list[str]] = {}
    missing = 0
    for deck_file in deck_files:
        for index, note in enumerate(deck_file.notes, start=1):
            if not note.note_key:
                missing += 1
                continue
            note_key_locations.setdefault(note.note_key, []).append(
                f"{deck_file.file_path.name} note {index}"
            )
    if missing:
        note_label = "note" if missing == 1 else "notes"
        raise ValueError(
            f"Missing note_keys for {missing} {note_label}. note_keys are stable "
            "IDs AnkiOps needs to match notes across collections without "
            "duplicates. Fix: run 'ankiops fa' to assign them."
        )
    duplicates = [
        f"Duplicate note_key '{note_key}' in {', '.join(locations)}"
        for note_key, locations in note_key_locations.items()
        if len(locations) > 1
    ]
    if duplicates:
        raise ValueError("; ".join(duplicates))
    return set(note_key_locations)


def _raise_source_parse_error(
    source: DeckSource,
    md_file: Path,
    error: Exception,
) -> None:
    identity = re.search(
        r"Unknown note type 'collab/([^/']+)/([^/']+)/",
        str(error),
    )
    if identity:
        try:
            canonical = parse_github_slug("/".join(identity.groups()))
        except ValueError:
            canonical = None
        if canonical and canonical != source.display_name:
            raise ValueError(
                f"This repository's deck files belong to {canonical}, not "
                f"{source.display_name}. Subscribe to the canonical deck instead: "
                f"ankiops collab subscribe {canonical}"
            ) from error
    raise ValueError(
        f"Error parsing {source.display_name}:{md_file.name}: {error}"
    ) from error


def _validate_parsed_sources(parsed_sources: tuple[ParsedSource, ...]) -> None:
    errors: list[str] = []
    deck_locations: dict[str, str] = {}
    note_key_locations: dict[str, str] = {}

    for parsed_source in parsed_sources:
        source_name = parsed_source.source.display_name
        for parsed_deck in parsed_source.decks:
            location = f"{source_name}:{parsed_deck.path.name}"
            previous_deck = deck_locations.get(parsed_deck.deck_name)
            if previous_deck is not None:
                errors.append(
                    f"Duplicate deck name '{parsed_deck.deck_name}' in "
                    f"{previous_deck} and {location}."
                )
            else:
                deck_locations[parsed_deck.deck_name] = location

            for index, note in enumerate(parsed_deck.parsed.notes, start=1):
                if note.note_key is None:
                    continue
                note_location = f"{location} note {index}"
                previous_note = note_key_locations.get(note.note_key)
                if previous_note is not None:
                    errors.append(
                        f"Duplicate note_key '{note.note_key}' in "
                        f"{previous_note} and {note_location}."
                    )
                else:
                    note_key_locations[note.note_key] = note_location

    if errors:
        raise ValueError("Invalid collection:\n- " + "\n- ".join(errors))


def _deck_matches_filter(
    deck_name: str,
    *,
    deck_filter: str | None,
    no_subdecks: bool,
) -> bool:
    if deck_filter is None:
        return True
    if no_subdecks:
        return deck_name == deck_filter
    return deck_name == deck_filter or deck_name.startswith(f"{deck_filter}::")


def serialize_to_file(
    collection_root: Path,
    output_file: Path,
    *,
    deck: str | None = None,
    no_subdecks: bool = False,
) -> dict[str, Any]:
    """Serialize markdown decks to a JSON file.

    Args:
        collection_root: Path to the markdown decks directory
        output_file: Path where JSON file will be written
        deck: Optional deck name scope for serialization, otherswise serializes the
              entire collection
        no_subdecks: If True with deck set, include only exact deck

    Returns:
        Dictionary containing the serialized data
    """
    serialized_data = serialize(
        collection_root,
        deck=deck,
        no_subdecks=no_subdecks,
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
    collection_root: Path,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Deserialize markdown decks from an in-memory JSON-compatible mapping."""
    logger.debug(f"Target directory: {collection_root}")

    db_path = collection_root / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an initialized AnkiOps collection: {collection_root}")

    decks = _validate_serialized_data(
        data,
        collection_root=collection_root,
    )
    _write_validated_decks(
        decks,
        collection_root=collection_root,
        overwrite=overwrite,
        quiet=quiet,
    )


def apply_deserialization_plan(
    plan: DeserializationPlan,
    *,
    collection_root: Path,
    overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """Write a previously validated deserialization plan."""
    logger.debug(f"Target directory: {collection_root}")

    db_path = collection_root / ANKIOPS_DB
    if not db_path.exists():
        raise ValueError(f"Not an initialized AnkiOps collection: {collection_root}")

    _write_validated_decks(
        plan.decks,
        collection_root=collection_root,
        overwrite=overwrite,
        quiet=quiet,
    )


def _write_validated_decks(
    decks: Sequence[_ValidatedDeck],
    *,
    collection_root: Path,
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
        f"{len({deck.source_name for deck in decks})} source(s) to {collection_root}"
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
    collection_root: Path,
) -> list[_ValidatedDeck]:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("Serialized data must be a JSON object mapping")

    raw_decks = data.get("decks")
    if not isinstance(raw_decks, list):
        raise ValueError("Serialized data must contain a top-level 'decks' list")

    sources = discover_deck_sources(collection_root)
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
    collection_root: Path,
) -> DeserializationPlan:
    with json_file.open("r", encoding="utf-8") as input_handle:
        data = json.load(input_handle)

    logger.debug(f"Importing serialized data from: {json_file}")
    decks = _validate_serialized_data(
        data,
        collection_root=collection_root,
    )
    return DeserializationPlan(
        decks=tuple(decks),
        target_paths=tuple(_deserialize_target_path(deck) for deck in decks),
        has_collab_sources=any(deck.source.is_collab for deck in decks),
    )


def deserialize_from_file(
    json_file: Path,
    overwrite: bool = False,
    *,
    collection_root: Path | None = None,
) -> None:
    """Deserialize markdown decks from a JSON file.

    Args:
        json_file: Path to JSON file to deserialize
        overwrite: If True, overwrite existing markdown files; if False, skip
    """
    collection_root = collection_root or get_collection_root()
    plan = plan_deserialize_from_file(
        json_file,
        collection_root=collection_root,
    )
    apply_deserialization_plan(
        plan,
        collection_root=collection_root,
        overwrite=overwrite,
    )

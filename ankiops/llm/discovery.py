"""Shared candidate discovery for LLM planning and execution."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ankiops.models import ANKIOPS_KEY_FIELD, NoteTypeConfig

from .task_types import FieldAccess, LlmCandidateStatus, NotePayload, TaskConfig


@dataclass(frozen=True)
class DiscoveryCounts:
    decks_seen: int
    decks_matched: int
    notes_seen: int


@dataclass(frozen=True)
class DiscoveryItem:
    ordinal: int
    deck_name: str
    note_key: str | None
    note_type: str | None
    candidate_status: LlmCandidateStatus
    skip_reason: str | None
    error_message: str | None
    payload: NotePayload | None
    note_type_config: NoteTypeConfig | None
    serialized_note: dict[str, Any] | None


@dataclass(frozen=True)
class DiscoverySnapshot:
    counts: DiscoveryCounts
    items: list[DiscoveryItem]


def iter_decks(data: dict[str, Any]) -> Iterator[tuple[str, list[Any]]]:
    decks = data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Serialized collection is missing a decks list")

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            continue

        yield deck_name, notes


def build_note_payload(
    task: TaskConfig,
    *,
    note: dict[str, Any],
    note_type_field_names: set[str],
) -> NotePayload | None:
    note_key = note.get("note_key")
    note_type = note.get("note_type")
    fields = note.get("fields")
    if (
        not isinstance(note_key, str)
        or not isinstance(note_type, str)
        or not isinstance(fields, dict)
    ):
        raise ValueError("Serialized note is missing note_key, note_type, or fields")

    editable_fields: dict[str, str] = {}
    read_only_fields: dict[str, str] = {}

    for field_name, raw_value in fields.items():
        if field_name == ANKIOPS_KEY_FIELD.name:
            continue
        if field_name not in note_type_field_names:
            continue
        if not isinstance(raw_value, str) or not raw_value:
            continue

        access = task.field_access(note_type, field_name)
        if access is FieldAccess.HIDDEN:
            continue
        if access is FieldAccess.READ_ONLY:
            read_only_fields[field_name] = raw_value
        else:
            editable_fields[field_name] = raw_value

    if not editable_fields:
        return None

    return NotePayload(
        note_key=note_key,
        note_type=note_type,
        editable_fields=editable_fields,
        read_only_fields=read_only_fields,
    )


def discover_candidates(
    *,
    data: dict[str, Any],
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
) -> DiscoverySnapshot:
    note_type_field_names = {
        name: {
            field.name
            for field in config.fields
            if field.name != ANKIOPS_KEY_FIELD.name
        }
        for name, config in note_type_configs.items()
    }
    items: list[DiscoveryItem] = []
    decks_seen = 0
    decks_matched = 0
    notes_seen = 0
    ordinal = 0

    for deck_name, notes in iter_decks(data):
        decks_seen += 1
        notes_seen += len(notes)
        in_scope = task.decks.matches(deck_name)
        if in_scope:
            decks_matched += 1

        for note in notes:
            if not isinstance(note, dict):
                continue
            ordinal += 1
            note_key = note.get("note_key")
            note_type_name = note.get("note_type")
            note_key_value = note_key if isinstance(note_key, str) else None
            note_type_value = (
                note_type_name if isinstance(note_type_name, str) else None
            )

            if not in_scope:
                items.append(
                    DiscoveryItem(
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key_value,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.SKIPPED_DECK_SCOPE,
                        skip_reason="outside task scope",
                        error_message=None,
                        payload=None,
                        note_type_config=None,
                        serialized_note=None,
                    )
                )
                continue

            note_type_config = (
                note_type_configs.get(note_type_name)
                if isinstance(note_type_name, str)
                else None
            )
            if note_type_config is None:
                message = f"Unknown note type '{note_type_name}' in serialized note"
                items.append(
                    DiscoveryItem(
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key_value,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.INVALID_NOTE,
                        skip_reason=None,
                        error_message=message,
                        payload=None,
                        note_type_config=None,
                        serialized_note=None,
                    )
                )
                continue

            try:
                payload = build_note_payload(
                    task,
                    note=note,
                    note_type_field_names=note_type_field_names[note_type_config.name],
                )
            except ValueError as error:
                items.append(
                    DiscoveryItem(
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key_value,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.INVALID_NOTE,
                        skip_reason=None,
                        error_message=str(error),
                        payload=None,
                        note_type_config=note_type_config,
                        serialized_note=note,
                    )
                )
                continue

            if payload is None:
                items.append(
                    DiscoveryItem(
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key_value,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS,
                        skip_reason="no editable non-empty fields",
                        error_message=None,
                        payload=None,
                        note_type_config=note_type_config,
                        serialized_note=note,
                    )
                )
                continue

            items.append(
                DiscoveryItem(
                    ordinal=ordinal,
                    deck_name=deck_name,
                    note_key=payload.note_key,
                    note_type=payload.note_type,
                    candidate_status=LlmCandidateStatus.ELIGIBLE,
                    skip_reason=None,
                    error_message=None,
                    payload=payload,
                    note_type_config=note_type_config,
                    serialized_note=note,
                )
            )

    return DiscoverySnapshot(
        counts=DiscoveryCounts(
            decks_seen=decks_seen,
            decks_matched=decks_matched,
            notes_seen=notes_seen,
        ),
        items=items,
    )

"""Task deck/note selection and payload extraction helpers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ankiops.ai.task_apply import add_warning
from ankiops.ai.types import InlineNotePayload, TaskConfig, TaskRunResult


@dataclass(frozen=True)
class GlobMatcher:
    patterns: tuple[str, ...]

    @classmethod
    def from_patterns(cls, patterns: list[str]) -> GlobMatcher:
        return cls(patterns=tuple(patterns))

    def matches(self, value: str) -> bool:
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in self.patterns)

    def select_names(self, names: Iterable[str]) -> list[str]:
        return [name for name in names if self.matches(name)]


@dataclass(frozen=True)
class TaskMatchers:
    note_types: GlobMatcher
    write_fields: GlobMatcher
    read_fields: GlobMatcher

    @classmethod
    def from_task(cls, task_config: TaskConfig) -> TaskMatchers:
        return cls(
            note_types=GlobMatcher.from_patterns(task_config.scope_note_types),
            write_fields=GlobMatcher.from_patterns(task_config.write_fields),
            read_fields=GlobMatcher.from_patterns(task_config.read_fields),
        )


@dataclass
class NoteTask:
    deck_name: str
    note_key: str
    note_type: str
    note_fields: dict[str, Any]
    original_write_fields: dict[str, str]
    write_fields: tuple[str, ...]
    payload: InlineNotePayload
    deck_index: int


def iter_note_tasks(
    *,
    selected_decks: list[dict[str, Any]],
    matchers: TaskMatchers,
    max_warnings: int,
    result: TaskRunResult,
) -> Iterator[NoteTask]:
    """Yield all note tasks matching task scope and available write fields."""
    for deck_index, deck in enumerate(selected_decks):
        deck_name = str(deck.get("name", ""))
        notes = deck.get("notes")
        if not isinstance(notes, list):
            continue

        result.processed_decks += 1
        for note in notes:
            if not isinstance(note, dict):
                continue
            result.processed_notes += 1

            note_key = normalize_note_key(note.get("note_key"))
            if note_key is None:
                add_warning(
                    result,
                    f"{deck_name}/<missing-note_key>: skipped note without note_key",
                    max_warnings=max_warnings,
                )
                continue

            note_type = normalize_note_type(note.get("note_type"))
            if note_type is None:
                add_warning(
                    result,
                    f"{deck_name}/{note_key}: skipped note without note_type",
                    max_warnings=max_warnings,
                )
                continue
            if not matchers.note_types.matches(note_type):
                continue

            fields = note.get("fields")
            if not isinstance(fields, dict):
                continue
            string_fields = {
                field_name: value
                for field_name, value in fields.items()
                if isinstance(field_name, str) and isinstance(value, str)
            }
            if not string_fields:
                continue

            write_fields = matchers.write_fields.select_names(string_fields.keys())
            if not write_fields:
                continue

            read_fields = merge_read_and_write_fields(
                read_fields=matchers.read_fields.select_names(string_fields.keys()),
                write_fields=write_fields,
            )
            payload_fields = {
                field_name: string_fields[field_name] for field_name in read_fields
            }
            write_snapshot = {
                field_name: string_fields[field_name] for field_name in write_fields
            }
            result.matched_notes += 1
            yield NoteTask(
                deck_name=deck_name,
                note_key=note_key,
                note_type=note_type,
                note_fields=fields,
                original_write_fields=write_snapshot,
                write_fields=tuple(write_fields),
                payload=InlineNotePayload(
                    note_key=note_key,
                    note_type=note_type,
                    fields=payload_fields,
                ),
                deck_index=deck_index,
            )


def require_decks(serialized_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Require and return top-level decks list."""
    decks = serialized_data.get("decks")
    if not isinstance(decks, list):
        from .errors import TaskExecutionError

        raise TaskExecutionError("serialized_data must contain 'decks' list")
    return decks


def select_decks(
    decks: list[dict[str, Any]],
    *,
    include_decks: list[str] | None,
    include_subdecks: bool,
) -> list[dict[str, Any]]:
    """Select decks by explicit names/globs with optional subdeck expansion."""
    targets = [name.strip() for name in (include_decks or []) if name.strip()]
    if not targets or "*" in targets:
        return decks
    return [
        deck
        for deck in decks
        if any(
            matches_deck_scope(str(deck.get("name", "")), target, include_subdecks)
            for target in targets
        )
    ]


def matches_deck_scope(deck_name: str, target: str, include_subdecks: bool) -> bool:
    """Match one deck name against one target (glob or concrete deck path)."""
    if any(char in target for char in ("*", "?", "[")):
        return fnmatch.fnmatchcase(deck_name, target)
    return deck_name == target or (
        include_subdecks and deck_name.startswith(f"{target}::")
    )


def normalize_note_key(raw_note_key: Any) -> str | None:
    if not isinstance(raw_note_key, str):
        return None
    note_key = raw_note_key.strip()
    return note_key or None


def normalize_note_type(raw_note_type: Any) -> str | None:
    if not isinstance(raw_note_type, str):
        return None
    note_type = raw_note_type.strip()
    return note_type or None


def merge_read_and_write_fields(
    *,
    read_fields: list[str],
    write_fields: list[str],
) -> list[str]:
    """Ensure write fields are always included in the request payload."""
    merged = list(read_fields)
    seen = set(merged)
    for field_name in write_fields:
        if field_name not in seen:
            merged.append(field_name)
            seen.add(field_name)
    return merged
